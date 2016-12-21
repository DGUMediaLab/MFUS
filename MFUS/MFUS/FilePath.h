#pragma once

#include <string>

//////////////////////////////////////////////////////////////////////////
//여기서 path 설정
static std::string filepath = "C:\\Users\\Jonha\\Desktop\\";
static std::string dataName = "Data11";
//////////////////////////////////////////////////////////////////////////

class FilePath{
private:
	FilePath(){}
	static FilePath* instance_;

public:
	static FilePath* getInstance();
	const std::string getColorPath(int number);
	const std::string getDepthPath(int number);
	const std::string getBodyIndexPath(int number);
};
