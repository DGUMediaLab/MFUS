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

void loadDepthImageFile(BYTE*& depthData, int frameNumber);

int main(){
	/**/printf("OpenCV Version : %s\n\n", CV_VERSION);

	BYTE* depthData = new BYTE[DEPTH_HEIGHT * DEPTH_WIDTH];
	loadDepthImageFile(depthData, 20);

	cv::Mat image(DEPTH_HEIGHT, DEPTH_WIDTH, CV_8UC1, cv::Scalar(0));
	for (int row = 0; row < DEPTH_HEIGHT; row++){
		for (int col = 0; col < DEPTH_WIDTH; col++){
			image.at<uchar>(row, col) = depthData[row * DEPTH_WIDTH + col];
		}
	}

	cv::imshow("Image", image);

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