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


//Data �о���� �޼ҵ��
void loadDepthImageFile(BYTE*& depthData, int frameNumber);
void loadBodyIndexFile(BYTE*& bodyIndexData, int frameNumber);

int main(){
	/**/printf("OpenCV Version : %s\n\n", CV_VERSION);

	BYTE* bodyIndexData;
	loadBodyIndexFile(bodyIndexData, 20);

	cv::Mat image(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC1, cv::Scalar(255));
	
	for (int row = 0; row < COLOR_HEIGHT; row++){
		for (int col = 0; col < COLOR_WIDTH * 2; col+=2){

			if (bodyIndexData[row * COLOR_WIDTH * 2 + col] < 6)
				image.at<uchar>(row, col / 2) = 0;
		}
	}

	cv::imshow("Image", image);

	cv::waitKey(0);

	delete bodyIndexData;

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