#include "FilePath.h"

FilePath* FilePath::instance_ = nullptr;

FilePath* FilePath::getInstance(){
	if (instance_ == nullptr)
		instance_ = new FilePath();

	return instance_;
}

const std::string FilePath::getColorPath(int number){

	return filepath + dataName + std::string("\\Color\\") + std::string("KinectScreenshot_RGB") + std::to_string(number) + std::string(".bmp");
}

const std::string FilePath::getDepthPath(int number){

	return filepath + dataName + std::string("\\Depth\\") + std::string("Filedepth_") + std::to_string(number) + std::string(".bin");
}

const std::string FilePath::getBodyIndexPath(int number){

	return filepath + dataName + std::string("\\HR_BodyIndex\\") + std::string("FileHRbodyIndex_") + std::to_string(number) + std::string(".bin");
}