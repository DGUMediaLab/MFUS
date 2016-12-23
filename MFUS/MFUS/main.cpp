//OpenCV 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

//�̸� ���� �� ��ġ �� Ÿ��
#include "Domains.h"
//data�� �а� ���� ���� ���� ��ġ
#include "FilePath.h"
//.bin ���̳ʸ� ������ �б� ����
#include "BinaryStream.h"


#include <time.h>


//Data �о���� �޼ҵ��
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


//����������
//���� ������׷�
double** accumulated_histogram;


//������ ���Ǵ� �Ķ���͵�

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

	//���� ������ ����
	//srand(time(NULL));
	
	for (int kFrameNumber = 0; kFrameNumber < 100; kFrameNumber++){

#pragma region init parameters
		//accumulated background histogram �ʱ�ȭ
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

		//�� �ܰ� ���� �� �ʿ��� �޸� ����
		delete bodyIndexData;
#pragma endregion

		/*
		3.2 Foreground Hole Detection
		*/
#pragma region Step : 3.2

		//�̹� frame(time t)���� ��� segmentation�� ����� ���� mat
		cv::Mat current_segmentation(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC1, cv::Scalar(BACKGROUND));

		for (int row = 0; row < COLOR_HEIGHT; row++){
			for (int col = 0; col < COLOR_WIDTH; col++){
				//�ʱ� ������� segmentation ��������� �̸� ����ִ´�.
				if (initial_segmentation[row * COLOR_WIDTH + col] == FOREGROUND){
					current_segmentation.at<uchar>(row, col) = FOREGROUND;
				}
			}
		}
		//cv::imshow("Initial Segmentation", img_initial_segmentation);

		//Initial binary image�κ��� ��� contours ���ϱ�
		//�ܰ��� �迭. ��, ������ �䱸�ϴ� ��� contours
		vector<vector<cv::Point>> contours;
		//�ܰ����� ���� ��������
		vector<cv::Vec4i> hierarchy;
		//CV_RETR_TREE = contour �˻� ����� tree ������ ����. �� �� �ٱ� �ʿ� ���� ���� ��Ʈ
		cv::findContours(current_segmentation, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);


		int nAreaCount = contours.size();
		printf("contour�� ���� = %d\n", nAreaCount);

		//�ȼ��� Countour�� �ִ� ��ġ���� �ƴ��� ������ Ȯ���ϱ� ���� ����
		cv::Mat isInContour(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC1, cv::Scalar(0));
		for (int ci = 0; ci < nAreaCount; ci++){
			for (int pi = 0; pi < contours[ci].size(); pi++){
				isInContour.at<uchar>(contours[ci][pi].y, contours[ci][pi].x) = 1;
			}
		}

		//�÷� ���� �о����
		cv::Mat img_input_color = cv::imread(FilePath::getInstance()->getColorPath(kFrameNumber));
		cv::Mat img_input_gray;

		cv::cvtColor(img_input_color, img_input_gray, CV_RGB2GRAY);
		
		const double kThreshold_hole = 0.5;

		//hole�� ������ŭ �ݺ�
		for (int ci = 0; ci < nAreaCount; ci++){

			//ci ��° contour�� ������ ĥ�� ��,
			cv::Mat img_contourArea(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC1, cv::Scalar(0));
			cv::drawContours(img_contourArea, contours, ci, cv::Scalar(255), -1, 8, hierarchy);
			//findNonZero���� 0�� �ƴ� �κ��� ã�Ƽ� contourPixels�� ����
			//��, contour(=hole) ���� ���� pixel ��ǥ�� ���� ����.
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

			//threshold value(0.5)���� ũ�� foreground hole�� ����.
			//if (p_foreground_hole > kThreshold_hole){
			if (0){
				//���θ� ä��� ���� ������ thickness�� �Է�(API Ȯ��)
				int thickness = -1;

				//foreground = 1
				cv::Scalar color = cv::Scalar(FOREGROUND);

				//���� ������ ���� ������ �����ϰ� ������ �Ʒ� �ּ��� �̿� img_initial_segmentation ��� 8UC3 Mat �ϳ� �����ؼ� �����ϸ� ��.
				//cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

				//hole�� ���θ� ä���.
				cv::drawContours(current_segmentation, contours, ci, color, thickness, 8, hierarchy);
			}
		}
		//cv::imshow("Contours", current_segmentation);		

		//���� ������׷� ����
		for (int row = 0; row < COLOR_HEIGHT; row++){
			for (int col = 0; col < COLOR_WIDTH; col++){

				//���� ���� pixel(z)�� background�� ���еǸ�, ���� ������׷��� �����Ѵ�.
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

		//���� �ܰ� ����	


		//������ �ܰ�
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
	loadBodyIndexFile�� ������ ���
	��, depth �������� ��� 1byte�� �ƴ϶� 2byte�̹Ƿ� 2byte �� �о�´�.
	*/
	const ushort minDepth = 500;	//�ʹ� ���� depth���� �ɷ����� ���� ��
	const ushort maxDepth = 65535;	//unsigned short's max value
	const int MapDepthToByte = 8000 / 256;

	BinaryReader br(FilePath::getInstance()->getDepthPath(frameNumber));

	int cur_pos = 0;
	int file_length = (int)DEPTH_HEIGHT * DEPTH_WIDTH;

	//HACK: ����Ʈ �����͸� ����ϴ� ��,
	//����� ������Ʈ�� �۾Ƽ� main ���������� delete�� ���� �޸� ������ ���� ��
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
	(1920 * 2) * 1080�� ����
	L1, R1, L2, R2, L3, R3, L4, R4 ... ������ ����
	L�� ���� ������ ��. ��, bodyIndex�� 0~5�������� ��ȣ�� �������
	R�� ���� �ش� ���� ��Ȯ���� ��Ÿ��. �� 0, 1 �� ���� ������ 0�̸� ������ ��, 1�̸� ��Ȯ�� ���� ��Ÿ����.
	*/
	BinaryReader br(FilePath::getInstance()->getBodyIndexPath(frameNumber));
	int cur_pos = 0;
	int file_length = (int)3840 * 1080;

	//HACK: ����Ʈ �����͸� ����ϴ� ��, ����� ������Ʈ�� �۾Ƽ� main ���������� delete�� ���� �޸� ������ ���� ��
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
			//���� ����� 1���̶�� ����
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
		//�ñ׸� h^l_z,t
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