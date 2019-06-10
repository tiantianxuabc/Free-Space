#include <iostream>
#include <fstream>
#include <opencv.hpp>
#include "freespace.h"

#ifdef _DEBUG
#pragma comment(lib,"opencv_world400d")
#else
#pragma comment(lib, "opencv_world400")
#endif

static void help()
{
	std::string hotkeys =
		"\n\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tt - toggle compute mode [MODE_DP/MODE_MIN]\n"
		"\tp - pause video\n";

	std::cout << hotkeys;
}

int main(int argc, char *argv[])
{

	const int numDisparities = 128;
	const int SADWindowSize = 11;
	//cv::StereoSGBM ssgbm;
	cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(0, 128, 3,11, 100, 1000,	32, 0, 15, 1000, 16);;

	
	

	// input camera parameters
	cv::FileStorage cvfs("camera.xml", cv::FileStorage::READ);
	if (!cvfs.isOpened())
	{
		std::cerr << "open camera.xml failed." << std::endl;
		return -1;
	}
	
	float fu = cvfs["FocalLengthX"];
	float fv = cvfs["FocalLengthY"];
	float u0 = cvfs["CenterX"];
	float v0 = cvfs["CenterY"];
	float baseline = cvfs["BaseLine"];
	float height = cvfs["Height"];
	float tilt = cvfs["Tilt"];

	FreeSpace freespace(fu, fv, u0, v0, baseline, height, tilt);
	int mode = FreeSpace::MODE_DP;

	std::string dir = "E:/Image Set/";
	//help();
	for (int frameno = 1;; frameno++)
	{
		char base_name[256];
		sprintf(base_name, "%06d.png", frameno);
		std::string bufl = dir + "1/training/0010/" + base_name;
		std::string bufr = dir + "2/training/0010/" + base_name;

		cv::Mat left = cv::imread(bufl, cv::IMREAD_GRAYSCALE);
		cv::Mat right = cv::imread(bufr, cv::IMREAD_GRAYSCALE);

		if (left.empty() || right.empty())
		{
			std::cerr << "imread failed." << std::endl;
			break;
		}

		CV_Assert(left.size() == right.size() && left.type() == right.type());

		switch (left.type())
		{
		case CV_8U:
			// nothing to do
			break;
		case CV_16U:
			// conver to CV_8U
			double maxval;
			cv::minMaxLoc(left, NULL, &maxval);
			left.convertTo(left, CV_8U, 255 / maxval);
			right.convertTo(right, CV_8U, 255 / maxval);
			break;
		default:
			std::cerr << "unsupported image type." << std::endl;
			return -1;
		}

		// calculate dispaliry
		cv::Mat disp;
		ssgbm->compute(left, right, disp);
		//ssgbm(left, right, disp);
		cv::Mat _dist;
		disp.convertTo(_dist, CV_8UC1);
		disp.convertTo(disp, CV_32F, 1.0 / 16);

		// calculate free space
		std::vector<int> bounds;
		freespace.compute(disp, bounds, 1, 1, mode);

		// draw free space
		cv::Mat draw;
		cv::cvtColor(left, draw, cv::COLOR_GRAY2BGRA);
		for (int u = 0; u < left.cols; u++)
			for (int v = bounds[u]; v < left.rows; v++)
				draw.at<cv::Vec4b>(v, u) += cv::Vec4b(0, 0, 255, 0);

		cv::imshow("free space", draw);
		cv::imshow("disparity", _dist);

		cv::Mat score;
		freespace.score_.convertTo(score, CV_8UC1);
		//cv::normalize(freespace.score_, freespace.score_, 0, 255, cv::NORM_MINMAX);
		cv::imshow("score", score);

		char c = cv::waitKey(1);
		if (c == 27)
			break;
		if (c == 'p')
			cv::waitKey(0);
		if (c == 't')
			mode = !mode;
	}

	return 0;
}