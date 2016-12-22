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
double getRegionColorLikelihood(vector< vector<cv::Point> > &contours, int contoursIndex, const cv::Mat& img_gray);
double getWeightRegionColorLikelihood();
double getContourSpatialPrior();
double getWeightContourSpatialPrior();


//3.2.1
double getColorLikelihood(cv::Point pos, const cv::Mat& img_gray);
double getWeightColorLikelihood();

//����������
//���� ������׷�
int** accumulated_histogram;


//������ ���Ǵ� �Ķ���͵�

//equation (6)
const int L = 32;
//equation (7)
const int C = 2;
//equation (8)
const double a = 1;
const double b = 0.95;



int main(){
	/**/printf("OpenCV Version : %s\n\n", CV_VERSION);

	//���� ������ ����
	//srand(time(NULL));

	const int kFrameNumber = 20;

#pragma region init parameters
	//accumulated background histogram �ʱ�ȭ
	accumulated_histogram = new int*[COLOR_HEIGHT * COLOR_WIDTH];

	for (int i = 0; i < COLOR_HEIGHT * COLOR_WIDTH; i++){
		accumulated_histogram[i] = new int[L];
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
	//initial Segmentation ���� �����, �������� ���
	cv::Mat img_initial_segmentation(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC1, cv::Scalar(0));
	for (int row = 0; row < COLOR_HEIGHT; row++){
		for (int col = 0; col < COLOR_WIDTH; col++){

			if (initial_segmentation[row * COLOR_WIDTH + col] == 1){
				img_initial_segmentation.at<uchar>(row, col) = 255;
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
	cv::findContours(img_initial_segmentation, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE); 


	int nAreaCount = contours.size();
	printf("contour�� ���� = %d\n", nAreaCount);	

	//�÷� ���� �о����
	cv::Mat img_input_color = cv::imread(FilePath::getInstance()->getColorPath(kFrameNumber));
	cv::Mat img_input_gray;

	cv::cvtColor(img_input_color, img_input_gray, CV_RGB2GRAY);

	const double kThreshold_hole = 0.5;
	
	//hole�� ������ŭ �ݺ�
	for (int ci = 0; ci < nAreaCount; ci++){
		
		//Step : 3.2.1 : Region's Color Likelihood and Its Weight
		double a_Region_Color_likelihood = getRegionColorLikelihood(contours, ci, img_input_gray);
		double w_Region_Color_likelihood = getWeightRegionColorLikelihood();

		//Step : 3.2.2 : Contour's Spatial Prior and Its Weight
		double a_Contour_Spatial_prior = getContourSpatialPrior();
		double w_Contour_Spatial_prior = getWeightContourSpatialPrior();

		//equation (2)
		double w_Dot_R_C = w_Region_Color_likelihood / (w_Region_Color_likelihood + w_Contour_Spatial_prior);

		//equation (2)
		double p_foreground_hole = w_Dot_R_C * a_Region_Color_likelihood + (1 - w_Dot_R_C) * a_Contour_Spatial_prior;

		//threshold value(0.5)���� ũ�� foreground hole�� ����.
		//if (p_foreground_hole > kThreshold_hole){
		if (1){						
			//���θ� ä��� ���� ������ thickness�� �Է�(API Ȯ��)
			int thickness = -1;

			//������ �������
			cv::Scalar color = cv::Scalar(255);
			//���� ������ ���� ������ �����ϰ� ������ �Ʒ� �ּ��� �̿� img_initial_segmentation ��� 8UC3 Mat �ϳ� �����ؼ� �����ϸ� ��.
			//cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

			//hole�� ���θ� ä���.
			cv::drawContours(img_initial_segmentation, contours, ci, color, thickness, 8, hierarchy);
		}
	}
	cv::imshow("Contours", img_initial_segmentation);
	
#pragma endregion

	//���� �ܰ� ����	


	//������ �ܰ�
	delete initial_segmentation;

	for (int i = 0; i < COLOR_HEIGHT * COLOR_WIDTH; i++){
		delete[] accumulated_histogram[i];
	}
	delete[] accumulated_histogram;

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

double getRegionColorLikelihood(vector< vector<cv::Point> > &contours, int ci, const cv::Mat& img_gray){
	/*
	3.2.1 Region's Color Likelihood and Its Weight

	a^rc = region_color_likelihood
	w^rc = w_region_color_likelihood

	a^cb = color_likelihood
	w^cb = w_color_likelihood
	w.^cb = norm_w_color_likelihood
	*/
	
	double sum_w_color_likelihood = 0;
	for (int pi = 0; pi < contours[ci].size(); pi++){

		double w_color_likelihood = getWeightColorLikelihood();
		//equation (4)
		sum_w_color_likelihood += w_color_likelihood;

	}

	double region_color_likelihood = 0;
	double w_region_color_likelihood = 0;

	for (int pi = 0; pi < contours[ci].size(); pi++){		

		double color_likelihood = getColorLikelihood(contours[ci][pi], img_gray);
		double w_color_likelihood = getWeightColorLikelihood();

		//equation (4)
		double norm_w_color_likelihood = w_color_likelihood / sum_w_color_likelihood;
		region_color_likelihood += norm_w_color_likelihood * color_likelihood;

		//equation (5)
		w_region_color_likelihood += w_color_likelihood;
	}
	//equation (5)
	w_region_color_likelihood / contours[ci].size();
	
	return 0;
}

double getWeightRegionColorLikelihood(){
	return 0;
}

double getContourSpatialPrior(){
	return 0;
}

double getWeightContourSpatialPrior(){
	return 0;
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

double getWeightColorLikelihood()
{
	return 0;
}