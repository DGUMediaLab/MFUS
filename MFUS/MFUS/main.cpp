//OpenCV 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include "opencv2/ml.hpp"


//미리 정의 된 수치 및 타입
#include "Domains.h"
//data를 읽고 쓰기 위한 파일 위치
#include "FilePath.h"
//.bin 바이너리 파일을 읽기 위함
#include "BinaryStream.h"


#include <time.h>


//Data 읽어오는 메소드들
void loadDepthImageFile(BYTE*& depthData, int frameNumber);
void loadBodyIndexFile(BYTE*& bodyIndexData, int frameNumber);


//3.1
void initialSegmentationMethod(BYTE*& initial_segmentation, BYTE* bodyIndexData);


//3.2
double getRegionColorLikelihood(const cv::Mat& contourPixels, int ci, const cv::Mat& img_gray);
double getWeightRegionColorLikelihood(const cv::Mat& contourPixels);
double getContourSpatialPrior(const cv::Mat& input_color, const cv::Mat& input_gray, BYTE*& initial_segmentation, const vector<cv::Point> &contours);
double getWeightContourSpatialPrior(const cv::Mat& input, const vector<cv::Point> &contours);

//3.2.1
double getColorLikelihood(cv::Point pos, const cv::Mat& img_gray);
double getWeightColorLikelihood(cv::Point pos);

//3.2.2
double getSpatialPrior(const cv::Mat& input_color, const cv::Mat& input_gray, BYTE*& initial_segmentation, const cv::Point pos);
double getWeightSpatialPrior(const cv::Mat& input, const cv::Point pos);
double getEdgeClarity(const cv::Mat& input_color, const cv::Mat& input_gray, BYTE*& initial_segmentation, const cv::Point pos);

//3.3
void performCoarseRefining(const cv::Mat& img_depth, const cv::Mat& img_gray, BYTE*& initial_segmentation);
bool getOmegaZero(BYTE*& initial_segmentation, const cv::Point pos);

//3.3.2
double getTemporalCue(BYTE*& previous_segmentation, cv::Point pos);
double getWeightTemporalCue(const cv::Mat& grad_x, const cv::Mat& grad_y, const cv::Mat& previous_grad_x, const cv::Mat& previous_grad_y, cv::Point pos);
double getColorCue(const cv::Mat& current_image, BYTE*& initial_segmentation, const cv::Point pos);
double getEdgeCue(BYTE*& initial_segmentation, const cv::Point pos);
double getWeightEdgeCue(BYTE*& previous_segmentation, BYTE*& initial_segmentation, const cv::Point pos);

//전역변수들
//누적 히스토그램
double** accumulated_histogram;


//논문에서 사용되는 파라미터들

//equation (6)
const int L = 32;
//equation (7)
const int C = 2;
//equation (8)
const double b = 0.95;

const int L_c = 7;
const int L_s = 11;

//FIXME : 정확한 수치를 아직 못찾음
const double gamma_F = 0.7;
const double gamma_B = 0.3;

const int L_b = 30;
const double tow_0 = 0.1;
const double tow_1 = 0.9;
const double tow_z = 0.5;
const double w_tow_z = 0.5;

const double sigma = 0.8;
const int L_e = 3;


/*
const int L_b = 7;
const double tow_0 = 0.3;
const double tow_1 = 0.7;
*/


int main(){
	/**/printf("OpenCV Version : %s\n\n", CV_VERSION);

	//난수 생성을 위해
	//srand(time(NULL));
	
	bool isInitPrevious_segmentation = false;
	cv::Mat previous_image = cv::imread(FilePath::getInstance()->getColorPath(0));
	BYTE* previous_final_segmentation = new BYTE[COLOR_HEIGHT * COLOR_WIDTH];
	cv::Mat previous_abs_grad_x, previous_abs_grad_y;
	for (int kFrameNumber = 20; kFrameNumber < 21; kFrameNumber++){

#pragma region init parameters
		//accumulated background histogram 초기화
		accumulated_histogram = new double*[COLOR_HEIGHT * COLOR_WIDTH];

		for (int i = 0; i < COLOR_HEIGHT * COLOR_WIDTH; i++){
			accumulated_histogram[i] = new double[L];
			for (int j = 0; j < L; j++){
				accumulated_histogram[i][j] = 0;
			}
		}

#pragma endregion

		/*
		3.1 Problem Formulation
		current time t = kFrameNumber
		x = inital_segmentation
		*/
#pragma region Step : 3.1
		BYTE* bodyIndexData;
		loadBodyIndexFile(bodyIndexData, kFrameNumber);

		//initial segmentation method
		BYTE* initial_segmentation;
		initialSegmentationMethod(initial_segmentation, bodyIndexData);

		if (isInitPrevious_segmentation == false){
			isInitPrevious_segmentation = true;
			memcpy(previous_final_segmentation, initial_segmentation, sizeof(initial_segmentation));
		}

		//각 단계 마다 불 필요한 메모리 삭제
		delete bodyIndexData;
#pragma endregion

		/*
		3.2 Foreground Hole Detection
		*/
#pragma region Step : 3.2

		//이번 frame(time t)에서 결과 segmentation의 결과를 담을 mat
		cv::Mat current_segmentation(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC1, cv::Scalar(BACKGROUND));

		for (int row = 0; row < COLOR_HEIGHT; row++){
			for (int col = 0; col < COLOR_WIDTH; col++){
				//초기 결과값을 segmentation 결과값으로 미리 집어넣는다.
				if (initial_segmentation[row * COLOR_WIDTH + col] == FOREGROUND){
					current_segmentation.at<uchar>(row, col) = FOREGROUND;
				}
			}
		}
		//cv::imshow("Initial Segmentation", current_segmentation);

		//Initial binary image로부터 모든 contours 구하기
		//외각선 배열. 즉, 논문에서 요구하는 모든 contours
		vector<vector<cv::Point>> contours;
		//외각선들 간의 계층구조
		vector<cv::Vec4i> hierarchy;
		//CV_RETR_TREE = contour 검색 결과를 tree 구조로 저장. 이 때 바깥 쪽에 있을 수록 루트
		cv::findContours(current_segmentation, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

		int nAreaCount = contours.size();
		printf("contour의 개수 = %d\n", nAreaCount);

		//픽셀이 Countour에 있는 위치인지 아닌지 빠르게 확인하기 위한 변수
		cv::Mat isInContour(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC1, cv::Scalar(0));
		for (int ci = 0; ci < nAreaCount; ci++){
			for (int pi = 0; pi < contours[ci].size(); pi++){
				isInContour.at<uchar>(contours[ci][pi].y, contours[ci][pi].x) = 1;
			}
		}

		//컬러 영상 읽어오기
		cv::Mat current_image = cv::imread(FilePath::getInstance()->getColorPath(kFrameNumber));
		//가우시안 블러
		cv::Mat current_gaussianBlur;
		//equation (22)
		cv::GaussianBlur(current_image, current_gaussianBlur, cv::Size(3, 3), sigma, sigma, cv::BORDER_DEFAULT);
		//color -> gray
		cv::Mat current_gray;
		cv::cvtColor(current_image, current_gray, CV_RGB2GRAY);

		cv::Mat current_gaussianblru_gray;
		cv::cvtColor(current_gaussianBlur, current_gaussianblru_gray, CV_RGB2GRAY);

		cv::Mat grad_x, grad_y;
		cv::Mat abs_grad_x, abs_grad_y;

		int scale = 1;
		int delta = 0;

		Sobel(current_gaussianBlur, grad_x, CV_16S, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel(current_gaussianBlur, grad_y, CV_16S, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
		
		const double kThreshold_hole = 0.5;

		//hole의 개수만큼 반복
		for (int ci = 0; ci < nAreaCount; ci++){

			//ci 번째 contour의 영역을 칠한 뒤,
			cv::Mat img_contourArea(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC1, cv::Scalar(0));
			cv::drawContours(img_contourArea, contours, ci, cv::Scalar(255), -1, 8, hierarchy);
			//findNonZero에서 0이 아닌 부분을 찾아서 contourPixels에 저장
			//즉, contour(=hole) 영역 안의 pixel 좌표만 따로 저장.
			//pixels set in the inner region of hole
			cv::Mat contourPixels;
			cv::findNonZero(img_contourArea, contourPixels);


			//Step : 3.2.1 : Region's Color Likelihood and Its Weight
			double a_Region_Color_likelihood = getRegionColorLikelihood(contourPixels, ci, current_gray);
			double w_Region_Color_likelihood = getWeightRegionColorLikelihood(contourPixels);

			//printf("%f %f\n", a_Region_Color_likelihood, w_Region_Color_likelihood);

			//Step : 3.2.2 : Contour's Spatial Prior and Its Weight
			double a_Contour_Spatial_prior = getContourSpatialPrior(current_image, current_gray, initial_segmentation, contours[ci]);
			double w_Contour_Spatial_prior = getWeightContourSpatialPrior(current_gray, contours[ci]);

			//equation (2)
			double w_Dot_R_C = w_Region_Color_likelihood / (w_Region_Color_likelihood + w_Contour_Spatial_prior);

			//equation (2)
			double p_foreground_hole = w_Dot_R_C * a_Region_Color_likelihood + (1 - w_Dot_R_C) * a_Contour_Spatial_prior;

			//threshold value(0.5)보다 크면 foreground hole로 인정.
			if (p_foreground_hole > kThreshold_hole){		
				//내부를 채우기 위해 음수의 thickness를 입력(API 확인)
				int thickness = -1;

				//foreground = 1
				cv::Scalar color = cv::Scalar(FOREGROUND);

				//만약 색상을 통해 눈으로 구분하고 싶으면 아래 주석을 이용 img_initial_segmentation 대신 8UC3 Mat 하나 생성해서 대입하면 됨.
				//cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

				//hole의 내부를 채운다.
				cv::drawContours(current_segmentation, contours, ci, color, thickness, 8, hierarchy);
			}
		}
	//	cv::imshow("Contours", current_segmentation);		
	//	cv::waitKey(0);
		//누적 히스토그램 갱신
		for (int row = 0; row < COLOR_HEIGHT; row++){
			for (int col = 0; col < COLOR_WIDTH; col++){

				//만약 현재 pixel(z)이 background로 구분되면, 누적 히스토그램을 갱신한다.
				if (current_segmentation.at<uchar>(row, col) == BACKGROUND){

					cv::Point pos(col, row);
										
					int delta = 0;
					
					if (initial_segmentation[row * COLOR_WIDTH + col] == BACKGROUND &&	//_z,t = 0
						isInContour.at<uchar>(row, col) == 0							//is not pixel(z) in boundary
						) {						
						delta = 1;
					}
					else{
						delta = 0;
					}
					int l = current_gray.at<uchar>(pos.y, pos.x) / 8;
					double a = 1 / (double)(kFrameNumber + 1);;

					//equation (8)
					double p_l = a * delta;
					accumulated_histogram[pos.y * COLOR_WIDTH + pos.x][l] = b * (1 - p_l) * accumulated_histogram[pos.y * COLOR_WIDTH + pos.x][l] + (1 - b * (1 - p_l));
					//printf("%f %f\n", p_l, accumulated_histogram[pos.y * COLOR_WIDTH + pos.x][l]);
				}
			}
		}

#pragma endregion


		/*
		3.3 Object Boundary Refining
		*/
		
#pragma region Step : 3.3

		//3.3.1
		//performCoarseRefining(img_input_gray, img_input_gray, initial_segmentation);


		BYTE* final_segmentation = new BYTE[COLOR_HEIGHT * COLOR_WIDTH];
		memcpy(final_segmentation, initial_segmentation, sizeof(initial_segmentation));

		//3.3.2
		for (int row = 0; row < COLOR_HEIGHT; row++){
			for (int col = 0; col < COLOR_WIDTH; col++){

				cv::Point pos(col, row);
				double a_temporal_cue = getTemporalCue(previous_final_segmentation, pos);
				//double a_edge_cue = getEdgeCue(initial_segmentation, pos);
				double a_color_cue = getColorCue(current_image, initial_segmentation, pos);
				
				double w_edge_cue = getWeightEdgeCue(previous_final_segmentation, initial_segmentation, pos);
				//double w_temporal_cue = getWeightTemporalCue(abs_grad_x, abs_grad_y, previous_abs_grad_x, previous_abs_grad_y, pos);
				double w_color_cue = getEdgeClarity(current_image, current_gray, initial_segmentation, pos);
								

				//equation (17) 3 left
				//double norm_w_temparal = w_temporal_cue / (w_temporal_cue + w_color_cue);

				//equation (17) 2 left
				//double a_temporal_color_cue = norm_w_temparal * a_temporal_cue + (1 - norm_w_temparal) * a_color_cue;

				//equation (17) 3 right
				//double w_temporal_color = w_temporal_cue > w_color_cue ? w_temporal_cue : w_color_cue;

				//equation (17) 2 right
				//double norm_w_temporal_color = w_temporal_color / (w_temporal_color + w_edge_cue);

				//equation (17) 1
				//double a_final_probability = norm_w_temporal_color * a_temporal_color_cue + (1 - norm_w_temporal_color) * a_edge_cue;


				//equation (18)
			//	if (a_final_probability > tow_z){
					//final_segmentation[row * COLOR_WIDTH + col] = 1;
				//}
			//	else
					//final_segmentation[row * COLOR_WIDTH + col] = 0;
			}
		}


#pragma endregion

		
		//이후 단계 구현	

		//마무리 단계
		delete initial_segmentation;

		for (int i = 0; i < COLOR_HEIGHT * COLOR_WIDTH; i++){
			delete[] accumulated_histogram[i];
		}
		delete[] accumulated_histogram;
		delete[] final_segmentation;

	}

	delete[] previous_final_segmentation;
	cv::waitKey(0);

	return 0;
}

void loadDepthImageFile(BYTE*& depthData, int frameNumber){
	/*
	loadBodyIndexFile와 동일한 방식
	단, depth 데이터의 경우 1byte가 아니라 2byte이므로 2byte 씩 읽어온다.
	*/
	const ushort minDepth = 500;	//너무 낮은 depth값은 걸러내기 위한 값
	const ushort maxDepth = 65535;	//unsigned short's max value
	const int MapDepthToByte = 8000 / 256;

	BinaryReader br(FilePath::getInstance()->getDepthPath(frameNumber));

	int cur_pos = 0;
	int file_length = (int)DEPTH_HEIGHT * DEPTH_WIDTH;

	//HACK: 스마트 포인터를 고려하는 중,
	//현재는 프로젝트가 작아서 main 마지막에서 delete를 통해 메모리 해제를 진행 중
	depthData = new BYTE[DEPTH_HEIGHT * DEPTH_WIDTH];

	int arr_index = 0;
	while (cur_pos < file_length)
	{
		ushort depthValue = br.ReadInt16();
		depthData[arr_index] = (depthValue >= minDepth && depthValue <= maxDepth ? (depthValue / MapDepthToByte) : 0);

		arr_index++;
		cur_pos++;
	}
}

void loadBodyIndexFile(BYTE*& bodyIndexData, int frameNumber){
	/*
	(1920 * 2) * 1080의 형태
	L1, R1, L2, R2, L3, R3, L4, R4 ... 순으로 저장
	L은 실제 데이터 값. 즉, bodyIndex의 0~5번까지의 번호가 들어있음
	R의 값은 해당 값의 정확도를 나타냄. 즉 0, 1 두 값만 가지며 0이면 보정된 값, 1이면 정확한 값을 나타낸다.
	*/
	BinaryReader br(FilePath::getInstance()->getBodyIndexPath(frameNumber));
	int cur_pos = 0;
	int file_length = (int)3840 * 1080;

	//HACK: 스마트 포인터를 고려하는 중, 현재는 프로젝트가 작아서 main 마지막에서 delete를 통해 메모리 해제를 진행 중
	bodyIndexData = new BYTE[3840 * 1080];

	int arr_index = 0;
	while (cur_pos < file_length)
	{
		bodyIndexData[arr_index] = br.ReadBYTE();

		arr_index++;
		cur_pos += sizeof(BYTE);
	}
}

void initialSegmentationMethod(BYTE*& initial_segmentation, BYTE* bodyIndexData){

	initial_segmentation = new BYTE[COLOR_HEIGHT * COLOR_WIDTH];

	for (int row = 0; row < COLOR_HEIGHT; row++){
		for (int col = 0; col < COLOR_WIDTH * 2; col += 2){
			//영상에 사람이 1명이라고 가정
			bodyIndexData[row * COLOR_WIDTH * 2 + col] < 6 ?
				initial_segmentation[row * COLOR_WIDTH + (col / 2)] = 1 :	//foreground
				initial_segmentation[row * COLOR_WIDTH + (col / 2)] = 0;	//background		
		}
	}
}

double getRegionColorLikelihood(const cv::Mat& contourPixels, int ci, const cv::Mat& img_gray){
	/*
	3.2.1 Region's Color Likelihood and Its Weight

	a^rc = region_color_likelihood

	a^cb = color_likelihood
	w^cb = w_color_likelihood
	w.^cb = norm_w_color_likelihood
	*/

	//equation (4) (right)
	double sum_w_color_likelihood = 0;
	for (int pi = 0; pi < contourPixels.total(); pi++){
		sum_w_color_likelihood += getWeightColorLikelihood(contourPixels.at<cv::Point>(pi));
	}

	double region_color_likelihood = 0;
	for (int pi = 0; pi <contourPixels.total(); pi++){

		double color_likelihood = getColorLikelihood(contourPixels.at<cv::Point>(pi), img_gray);
		double w_color_likelihood = getWeightColorLikelihood(contourPixels.at<cv::Point>(pi));


		//equation (4) (right)
		double norm_w_color_likelihood = w_color_likelihood / sum_w_color_likelihood;
		//equation (4) (left)		
		region_color_likelihood += norm_w_color_likelihood * color_likelihood;

	}

		/*
		for (int pi = 0; pi < contours[ci].size(); pi++){

		double color_likelihood = getColorLikelihood(contours[ci][pi], img_gray);
		double w_color_likelihood = getWeightColorLikelihood(contourPixels.at<cv::Point>(i));


		//equation (4) (right)
		double norm_w_color_likelihood = w_color_likelihood / sum_w_color_likelihood;
		//equation (4) (left)
		region_color_likelihood += norm_w_color_likelihood * color_likelihood;

		}
		*/
	
	return region_color_likelihood;
}

double getWeightRegionColorLikelihood(const cv::Mat& contourPixels){
	/*
	3.2.1 Region's Color Likelihood and Its Weight

	w^rc = w_region_color_likelihood
	w^cb = w_color_likelihood
	*/

	//equation (5)	
	double M = contourPixels.total();
	double sum_w_color_likelihood = 0;
	for (int i = 0; i < M; i++){
		sum_w_color_likelihood += getWeightColorLikelihood(contourPixels.at<cv::Point>(i));
	}

	double w_region_color_likelihood = sum_w_color_likelihood / M;


	return w_region_color_likelihood;
}

double getContourSpatialPrior(const cv::Mat& input_color, const cv::Mat& input_gray, BYTE*& initial_segmentation, const vector<cv::Point> &contours){
	/*
	3.2.2 Contour's Spatial Prior and Its Weight
	*/

	//equation (10) (right)
	double sum_w_spatial_prior = 0;
	for (int ci = 0; ci < contours.size(); ci++){
		sum_w_spatial_prior += getWeightSpatialPrior(input_gray, contours[ci]);
	}

	double a_Contour_Spatial_prior = 0;
	for (int ci = 0; ci < contours.size(); ci++){		

		double spatial_prior = getSpatialPrior(input_color, input_gray, initial_segmentation, contours[ci]);
		double w_spatial_prior = getWeightSpatialPrior(input_gray, contours[ci]);

		//equation (10) (right)
		double norm_w_spatial_prior = w_spatial_prior / sum_w_spatial_prior;
		//equation (10) (left)
		a_Contour_Spatial_prior += norm_w_spatial_prior * spatial_prior;
	}

	return a_Contour_Spatial_prior;
}

double getWeightContourSpatialPrior(const cv::Mat& input, const vector<cv::Point> &contours){
	/*
	3.2.2 Contour's Spatial Prior and Its Weight
	*/

	double N = contours.size();
	double sum_w_spatial_prior = 0;

	for (int ci = 0; ci < N; ci++){
		sum_w_spatial_prior += getWeightSpatialPrior(input, contours[ci]);
	}

	//equation (11)
	double w_Contour_Spatial_prior = sum_w_spatial_prior / N;

	return w_Contour_Spatial_prior;
}

double getColorLikelihood(cv::Point pos, const cv::Mat& img_gray){
	
	double a_color_likelihood = 0; 

	//p(V_z,t | y = 0)
	double p = 0;

	int l = img_gray.at<uchar>(pos.y, pos.x) / 8;
	for (int index = l - C; index < l + C; index++){
		//시그마 h^l_z,t
		p += accumulated_histogram[pos.y * COLOR_WIDTH + pos.x][l];
	}

	//equation (7)
	a_color_likelihood = 1 - p;

	return a_color_likelihood;
}

double getWeightColorLikelihood(cv::Point pos)
{
	//equation (9)
	double sigma_delta = 0;

	for (int i = 0; i < L; i++){
		if (accumulated_histogram[pos.y * COLOR_WIDTH + pos.x][i] != 0)
			sigma_delta++;
	}

	double w_color_likelihood = 1 - ( (sigma_delta - 1) / L);

	return w_color_likelihood;
}

double getSpatialPrior(const cv::Mat& input_color, const cv::Mat& input_gray, BYTE*& initial_segmentation, const cv::Point pos){

	double edge_clarity = getEdgeClarity(input_color, input_gray, initial_segmentation, pos);


	double spatial_prior = 0;

	//equation (13)
	if (1 - edge_clarity > gamma_F){
		spatial_prior = 1;
	}
	else if (1 - edge_clarity < gamma_B){
		spatial_prior = 0;
	}
	else{
		spatial_prior = (1 - edge_clarity - gamma_B) / (gamma_F - gamma_B);
	}

	//printf("%f\n", spatial_prior);
	return spatial_prior;
}

double getWeightSpatialPrior(const cv::Mat& input, const cv::Point pos){
	int *sl = new int[L];
	for (int i = 0; i < L; i++){
		sl[i] = 0;
	}


	int start_y = pos.y - L_c / 2;
	int end_y = pos.y + L_c / 2;

	int start_x = pos.x - L_c / 2;
	int end_x = pos.x + L_c / 2;

	if (start_y < 0) start_y = 0;
	if (end_y > COLOR_HEIGHT) end_y = COLOR_HEIGHT;

	if (start_x < 0) start_x = 0;
	if (end_x > COLOR_WIDTH) end_x = COLOR_WIDTH;

	for (int row = start_y; row < end_y; row++){
		for (int col = start_x; col < end_x; col++){
			sl[input.at<uchar>(row, col) / L]++;
		}
	}

	double sum = 0;
	for (int i = 0; i < L; i++){
		if (sl[i] != 0)
			sum++;
	}

	//equation (14)
	double w_spatial_prior = 1 - (sum - 1) / L;

	//printf("%f\n", w_spatial_prior);

	delete[] sl;
	return w_spatial_prior;
}

double getEdgeClarity(const cv::Mat& input_color, const cv::Mat& input_gray, BYTE*& initial_segmentation,  const cv::Point pos){

	double edge_clarity = 0;

	//double *sf = new double[L];
	//double *sb = new double[L];
	
	int *sl = new int[L];
	for (int i = 0; i < L; i++){
		sl[i] = 0;
	}


	int start_y = pos.y - L_c / 2;
	int end_y = pos.y + L_c / 2;

	int start_x = pos.x - L_c / 2;
	int end_x = pos.x + L_c / 2;

	if (start_y < 0) start_y = 0;
	if (end_y > COLOR_HEIGHT) end_y = COLOR_HEIGHT;

	if (start_x < 0) start_x = 0;
	if (end_x > COLOR_WIDTH) end_x = COLOR_WIDTH;

	cv::Mat foreground(L_c, L_c, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat background(L_c, L_c, CV_8UC3, cv::Scalar(0, 0, 0));

	//printf("%d\n", L_c / 2);
	int x = 0, y = 0;
	for (int row = start_y; row < end_y; row++, y++){
		for (int col = start_x; col < end_x; col++, x++){
			if (initial_segmentation[row * COLOR_WIDTH + col] == BACKGROUND){
				background.at<cv::Vec3b>(y, x) = input_color.at<cv::Vec3b>(row, col);
			
			}

			if (initial_segmentation[row * COLOR_WIDTH + col] == FOREGROUND){
				foreground.at<cv::Vec3b>(y, x) = input_color.at<cv::Vec3b>(row, col);
			}
			x = 0;
		}
	}

	cv::Mat re_foreground, re_background;

	cv::resize(background, re_background, cv::Size(350, 350), 0, 0, cv::INTER_NEAREST);
	cv::resize(foreground, re_foreground, cv::Size(350, 350), 0, 0, cv::INTER_NEAREST);

	//cv::imshow("Ddd", re_background);
	//cv::imshow("Ddd2", re_foreground);
	//cv::waitKey(0);

	/*
	for (int row = start_y; row < end_y; row++){
		for (int col = start_x; col < end_x; col++){
			sl[input.at<uchar>(row, col) / L]++;
		}
	}

	

	int count = 0;
	int first = 0;
	int second = 0;

	for (int i = 0; i < L; i++){
		if (count < sl[i]){
			second = first;
			first = i;
			count = sl[i];
		}
	}

	
	
	int N_z = 0;

	for (int i = 0; i < L; i++){
		if (i != first && i != second){
			N_z += sl[i];
		}
	}

	*/
	//edge_clarity = 1 - ((double)N_z / pow(L_c, 2));
	
	//printf("%f\n", edge_clarity);

	delete[] sl;

	/*/
	cv::Mat fgd_result, bgd_result;
	compare(current_segmentation, FOREGROUND, fgd_result, cv::CMP_EQ);
	img_input_color.copyTo(fgd_result, fgd_result);
	compare(current_segmentation, BACKGROUND, bgd_result, cv::CMP_EQ);
	img_input_color.copyTo(bgd_result, bgd_result);
	//cv::imshow("fgd", fgd_result);
	//cv::imshow("bgd", bgd_result);
	//cv::waitKey(0);



	vector<cv::Mat> bgr_planes;
	cv::split(fgd_result, bgr_planes);

	int histSize = 256;

	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	cv::Mat b_hist, g_hist, r_hist;

	cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	int hist_w = 700; int hist_h = 500;
	int bin_w = cvRound((double)hist_w / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for (int i = 1; i < histSize; i++){
		cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
		cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
		cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("dafdafd", histImage);
	cv::waitKey(0);
	*/
	//delete[] sf;
	//delete[] sb;
	

	return edge_clarity;
}

void performCoarseRefining(const cv::Mat& img_depth, const cv::Mat& img_gray, BYTE*& initial_segmentation){
	for (int row = 0; row < COLOR_HEIGHT; row++){
		for (int col = 0; col < COLOR_WIDTH; col++){

			//D = 0
			if (img_depth.at<uchar>(row, col) == 0){
				cv::Point pos(col, row);
				//z (= omega_0
				if (getOmegaZero(initial_segmentation, pos)){

					double a_color_likelihood = getColorLikelihood(pos, img_gray);
					double w_color_likelihood = getWeightColorLikelihood(pos);

					if (a_color_likelihood > tow_z && w_color_likelihood > w_tow_z){
						initial_segmentation[row * COLOR_WIDTH + col] = 1;
					}
					else{
						initial_segmentation = 0;
					}
				}
			}
		}
	}
}

bool getOmegaZero(BYTE*& initial_segmentation, const cv::Point pos){
	
	int start_y = pos.y - L_b / 2;
	int end_y = pos.y + L_b / 2;

	int start_x = pos.x - L_b / 2;
	int end_x = pos.x + L_b / 2;

	if (start_y < 0) start_y = 0;
	if (end_y > COLOR_HEIGHT) end_y = COLOR_HEIGHT;

	if (start_x < 0) start_x = 0;
	if (end_x > COLOR_WIDTH) end_x = COLOR_WIDTH;

	double sum_segmentation = 0;
	for (int row = start_y; row < end_y; row++){
		for (int col = start_x; col < end_x; col++){
			sum_segmentation += initial_segmentation[row * COLOR_WIDTH + col];
		}
	}
	//equation (15) right
	sum_segmentation /= powf(L_b, 2);

	//equation (15) left
	if (tow_0 < sum_segmentation < tow_1) return true;
	else return false;
}

double getTemporalCue(BYTE*& previous_segmentation, cv::Point pos){

	//equation (19)
	double a_temporal_cue = previous_segmentation[pos.y * COLOR_WIDTH + pos.x];

	return a_temporal_cue;
}
double getWeightTemporalCue(const cv::Mat& grad_x, const cv::Mat& grad_y, const cv::Mat& previous_grad_x, const cv::Mat& previous_grad_y, cv::Point pos){
	

	double m_z = 0;
	double total_m_z = 0;
	//equation (20)
	for (int y = pos.y - L_b / 2; y < pos.y + L_b / 2; y++){
		for (int x = pos.x - L_b / 2; x < pos.x + L_b / 2; x++){
			if (y >= 0 && y < COLOR_HEIGHT && x >= 0 && x < COLOR_WIDTH){
				double eH = grad_x.at<uchar>(x, y);
				double eV = grad_y.at<uchar>(x, y);

				double previous_eH = previous_grad_x.at<uchar>(x, y);
				double previous_eV = previous_grad_y.at<uchar>(x, y);

				//equation (21)
				double abs_T = sqrt(powf(eH, 2) + powf(eV, 2));

				m_z += abs_T * sqrt(powf(eH - previous_eH, 2) + powf(eV - previous_eV, 2));
				total_m_z += abs_T;
			}
		}
	}
	m_z /= total_m_z;

	//myu = <m_Z> + 3 <<m_z>>
	double myu = 0;

	//equation (23)
	double m_dot_z = 1 / (double)(1 + exp(-(m_z - myu)));

	//equation (24)
	double w_temporal_cue = 1 - m_dot_z;

	return w_temporal_cue;
}
double getColorCue(const cv::Mat& current_image, BYTE*& initial_segmentation, const cv::Point pos){
		
	int index_foreground = 0;
	int index_background = 0;
	for (int y = pos.y - L_s / 2; y <= pos.y + L_s / 2; y++){
		for (int x = pos.x - L_s / 2; x <= pos.x + L_s / 2; x++){
			if (x >= 0 && x < COLOR_WIDTH && y >= 0 && y < COLOR_HEIGHT){
				if (initial_segmentation[y * COLOR_WIDTH + x] == 1){
					index_foreground++;
				}
				else
					index_background++;
			}
		}
	}

	cv::Mat foreground_samples(index_foreground, 3, CV_32FC1);
	cv::Mat background_samples(index_background, 3, CV_32FC1);

	int a = 0, b = 0;
	for (int y = pos.y - L_s / 2; y <= pos.y + L_s / 2; y++){
		for (int x = pos.x - L_s / 2; x <= pos.x + L_s / 2; x++){
			if (x >= 0 && x < COLOR_WIDTH && y >= 0 && y < COLOR_HEIGHT
				){
				if (initial_segmentation[y * COLOR_WIDTH + x] == 1){
					foreground_samples.at<cv::Vec3f>(b, 0)[0] = current_image.at<cv::Vec3b>(y, x)[0];
					foreground_samples.at<cv::Vec3f>(b, 0)[1] = current_image.at<cv::Vec3b>(y, x)[1];
					foreground_samples.at<cv::Vec3f>(b++, 0)[2] = current_image.at<cv::Vec3b>(y, x)[2];
				}
				else{
					background_samples.at<cv::Vec3f>(b, 0)[0] = current_image.at<cv::Vec3b>(y, x)[0];
					background_samples.at<cv::Vec3f>(b, 0)[1] = current_image.at<cv::Vec3b>(y, x)[1];
					background_samples.at<cv::Vec3f>(b++, 0)[2] = current_image.at<cv::Vec3b>(y, x)[2];
				}
			}
		}
	}

	//default, nclusters = 5, EM::COV_MAT_DIAGONAL
	cv::EM foreground_EM, background_EM;
	if (index_foreground != 0)
		foreground_EM.train(foreground_samples);
	if (index_background != 0)
		background_EM.train(background_samples);

	

	

	cv::Mat sample(1, 3, CV_32FC1);
	sample.at<cv::Vec3f>(0, 0)[0] = current_image.at<cv::Vec3b>(pos.y, pos.x)[0];
	sample.at<cv::Vec3f>(0, 0)[1] = current_image.at<cv::Vec3b>(pos.y, pos.x)[1];
	sample.at<cv::Vec3f>(0, 0)[2] = current_image.at<cv::Vec3b>(pos.y, pos.x)[2];

	
	//printf("%d  %d\n",  predict[0], predict[1]);
	//printf("%f  %f %f %f %f  \n", gmm.at<long float>(0), gmm.at<long float>(1), gmm.at<long float>(2), gmm.at<long float>(3), gmm.at<long float>(4));

	double p_foreground = 0, p_background = 0;
	//equation (25)
	cv::Mat gmm;	
	//[0] likelihood, [1] index of mixture
	cv::Vec2b predict;
	if (index_foreground != 0){
		predict = foreground_EM.predict(sample, gmm);
		p_foreground = gmm.at<long float>(predict[1]);
	}
	if (index_background != 0){
		predict = background_EM.predict(sample, gmm);
		p_background = gmm.at<long float>(predict[1]);
	}

	double a_color_cue;

	//equation (26)
	a_color_cue = p_foreground / (p_foreground + p_background);
	
	printf("%f\n", a_color_cue);

	return a_color_cue;
}

double getEdgeCue(BYTE*& initial_segmentation, const cv::Point pos){

	double a_z = 0;
	double b_z = 0;
	double c_z = 0;
	double s_z = 0;

	double delta_z = 0;

	for (int y = pos.y - L_b / 2; y <= pos.y + L_b / 2; y++){
		for (int x = pos.x - L_b / 2; x <= pos.x + L_b / 2; x++){
			if (x >= 0 && x < COLOR_WIDTH && y >= 0 && y < COLOR_HEIGHT){
				s_z += initial_segmentation[y * COLOR_WIDTH + x];
			}
		}
	}
	//equation (15) right
	s_z /= powf(L_b, 2);

	//equation (28)
	double a_edge_cue = a_z / exp(-(s_z - c_z) / delta_z) + b_z;
	
	return a_edge_cue;
}
double getWeightEdgeCue(BYTE*& previous_segmentation, BYTE*& initial_segmentation, const cv::Point pos){
	
	//equation (29)
	double w_edge_cue = 0;

	for (int y = pos.y - L_e / 2; y <= pos.y + L_e / 2; y++){
		for (int x = pos.x - L_e / 2; x <= pos.x + L_e / 2; x++){
			if (x >= 0 && x < COLOR_WIDTH && y >= 0 && y < COLOR_HEIGHT){
				cv::Point neighborhoodPos(x, y);
				w_edge_cue += abs(getTemporalCue(previous_segmentation, neighborhoodPos) - initial_segmentation[y * COLOR_WIDTH + x]);
			}
		}
	}

	w_edge_cue /= powf(L_e, 2);

	return w_edge_cue;
}