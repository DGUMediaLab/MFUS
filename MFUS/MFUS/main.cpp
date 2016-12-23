//OpenCV 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

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
double getContourSpatialPrior(const vector<cv::Point> &contours);
double getWeightContourSpatialPrior(const vector<cv::Point> &contours);

//3.2.1
double getColorLikelihood(cv::Point pos, const cv::Mat& img_gray);
double getWeightColorLikelihood(cv::Point pos);

//3.2.2
double getSpatialPrior();
double getWeightSpatialPrior();


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

const int L_c = 6;
const int L_s = 11;

const int L_b = 30;
const double tow_0 = 0.1;
const double tow_1 = 0.9;

/*
const int L_b = 7;
const double tow_0 = 0.3;
const double tow_1 = 0.7;
*/


int main(){
	/**/printf("OpenCV Version : %s\n\n", CV_VERSION);

	//난수 생성을 위해
	//srand(time(NULL));
	
	for (int kFrameNumber = 0; kFrameNumber < 100; kFrameNumber++){

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
		//cv::imshow("Initial Segmentation", img_initial_segmentation);

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
		cv::Mat img_input_color = cv::imread(FilePath::getInstance()->getColorPath(kFrameNumber));
		cv::Mat img_input_gray;

		cv::cvtColor(img_input_color, img_input_gray, CV_RGB2GRAY);
		
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
			double a_Region_Color_likelihood = getRegionColorLikelihood(contourPixels, ci, img_input_gray);
			double w_Region_Color_likelihood = getWeightRegionColorLikelihood(contourPixels);

			//printf("%f %f\n", a_Region_Color_likelihood, w_Region_Color_likelihood);

			//Step : 3.2.2 : Contour's Spatial Prior and Its Weight
			double a_Contour_Spatial_prior = getContourSpatialPrior(contours[ci]);
			double w_Contour_Spatial_prior = getWeightContourSpatialPrior(contours[ci]);

			//equation (2)
			double w_Dot_R_C = w_Region_Color_likelihood / (w_Region_Color_likelihood + w_Contour_Spatial_prior);

			//equation (2)
			double p_foreground_hole = w_Dot_R_C * a_Region_Color_likelihood + (1 - w_Dot_R_C) * a_Contour_Spatial_prior;

			//threshold value(0.5)보다 크면 foreground hole로 인정.
			//if (p_foreground_hole > kThreshold_hole){
			if (0){
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
		//cv::imshow("Contours", current_segmentation);		

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
					int l = img_input_gray.at<uchar>(pos.y, pos.x) / 8;
					double a = 1 / (double)(kFrameNumber + 1);;

					//equation (8)
					double p_l = a * delta;
					accumulated_histogram[pos.y * COLOR_WIDTH + pos.x][l] = b * (1 - p_l) * accumulated_histogram[pos.y * COLOR_WIDTH + pos.x][l] + (1 - b * (1 - p_l));
					//printf("%f %f\n", p_l, accumulated_histogram[pos.y * COLOR_WIDTH + pos.x][l]);
				}
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

	}

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

double getContourSpatialPrior(const vector<cv::Point> &contours){
	/*
	3.2.2 Contour's Spatial Prior and Its Weight
	*/

	//equation (10) (right)
	double sum_w_spatial_prior = 0;
	for (int ci = 0; ci < contours.size(); ci++){
		sum_w_spatial_prior += getWeightSpatialPrior();
	}

	double a_Contour_Spatial_prior = 0;
	for (int ci = 0; ci < contours.size(); ci++){		

		double spatial_prior = getSpatialPrior();
		double w_spatial_prior = getWeightSpatialPrior();

		//equation (10) (right)
		double norm_w_spatial_prior = w_spatial_prior / sum_w_spatial_prior;
		//equation (10) (left)
		a_Contour_Spatial_prior += norm_w_spatial_prior * spatial_prior;
	}

	return a_Contour_Spatial_prior;
}

double getWeightContourSpatialPrior(const vector<cv::Point> &contours){
	/*
	3.2.2 Contour's Spatial Prior and Its Weight
	*/

	double N = contours.size();
	double sum_w_spatial_prior = 0;

	for (int i = 0; i < N; i++){
		sum_w_spatial_prior += getWeightSpatialPrior();
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

double getSpatialPrior(){
	return 0;
}

double getWeightSpatialPrior(){
	return 0;
}