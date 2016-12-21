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




//Data 읽어오는 메소드들
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
	I = colorImage
	D = depthImage
	x = inital_segmentation	
	*/		

	BYTE* bodyIndexData;	
	loadBodyIndexFile(bodyIndexData, kFrameNumber);

	//initial segmentation method
	BYTE* initial_segmentation;
	initialSegmentationMethod(initial_segmentation, bodyIndexData);

	//각 단계 마다 불 필요한 메모리 삭제
	delete bodyIndexData;


	//이후 단계 구현


	//마무리 단계
	delete initial_segmentation;

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