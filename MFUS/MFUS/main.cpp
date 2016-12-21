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


int main(){
	/**/printf("OpenCV Version : %s\n\n", CV_VERSION);

	const int kFrameNumber = 20;

	/*
	3.1 Problem Formulation
	current time t = kFrameNumber
	x = inital_segmentation	
	*/		

	BYTE* bodyIndexData;	
	loadBodyIndexFile(bodyIndexData, kFrameNumber);

	//initial segmentation method
	BYTE* initial_segmentation;
	initialSegmentationMethod(initial_segmentation, bodyIndexData);

	//�� �ܰ� ���� �� �ʿ��� �޸� ����
	delete bodyIndexData;

	/*
	3.2 Foreground Hole Detection
	*/

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

	//contours ���� �����	
	/* 
	srand(time(NULL));
	cv::Mat output(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < nAreaCount; i++){
		cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
		int thickness = 2;
		cv::drawContours(output, contours, i, color, thickness, 8, hierarchy);
	}
	cv::imshow("Contours", output);
	*/
	
	//���� �ܰ� ����


	//������ �ܰ�
	delete initial_segmentation;

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