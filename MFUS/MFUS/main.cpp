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