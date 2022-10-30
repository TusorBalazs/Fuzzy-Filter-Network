#include <fstream>
#include <iostream>
#include <string>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>
#include <time.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "FuzzyFilterNetwork.h"

using namespace cv;
using namespace std;

int main()
{
	//create the classifier
	FuzzyFilterNetwork * ffn = new FuzzyFilterNetwork(3);

	//uncomment the desired color space
		//colorspace col = HSV;
		colorspace col = LAB;
		//colorspace col = LUV;

	double limit = 15; // used for the clustering: the maximum allowed distance between two datapoints
	int TH0 = 10; //used for the evaluation: Theta threshold
	string croppedfilename = "hsvs_cropped.dat"; // contains the reduced dataset (without redundancy and inconsistencies)
	string rulefile = "rbf_rules.dat"; //the rulefile's name

	//create the output image file name, based on the parameters
	string outImage = "ffn_-r_";
	outImage += std::to_string((int)(limit * 100)); outImage += "_TH"; outImage += std::to_string(TH0);
	switch (col)
	{
	case RGB:
		outImage += "_RGB.png";
		break;
	case HSV:
		outImage += "_HSV.png";
		break;
	case LAB:
		outImage += "_LAB.png";
		break;
	case LUV:
		outImage += "_LUV.png";
		break;
	}
	double TH = (double)TH0 / 100; // turn the integer threshold into percentage

	//training image file locations & names
	string trainImage = "images/WL00060.jpg";
	string trainMask = "images/WL00060mask.png";

	Mat trainimg = imread(trainImage, cv::IMREAD_COLOR);  if (trainimg.empty()) { cout << "\n training image not found!"; return 0; }
	Mat trainmask = imread(trainMask, cv::IMREAD_COLOR);  if (trainmask.empty()) { cout << "\n training mask not found!"; return 0; }

	//testing image file locations & names
	string testImage = "images/WL00063.jpg";
	string testMask = "images/WL00063mask2.png";

	Mat testimg = imread(testImage, cv::IMREAD_COLOR); if (testimg.empty()) { cout << "\n testing image not found!"; return 0; }
	Mat testmask = imread(testMask, cv::IMREAD_COLOR); if (testmask.empty()) { cout << "\n testing mask not found!"; return 0; }

	// Make training data from a training image and the corresponding mask image
	ffn->MakeTrainingData(trainImage, trainMask, croppedfilename, col);

	// Train the classifier, i.e. cluster the training data
	ffn->TrainFFN(croppedfilename, rulefile, limit);

	// Make the logfile name
	string statsfile = "stats_";
	statsfile += std::to_string((int)(limit * 100)); statsfile += "_TH"; statsfile += std::to_string(TH0);
	switch (col)
	{
	case RGB:
		statsfile += "_RGB.txt";
		break;
	case HSV:
		statsfile += "_HSV.txt";
		break;
	case LAB:
		statsfile += "_LAB.txt";
		break;
	case LUV:
		statsfile += "_LUV.txt";
		break;
	}
	
	//evaluate with comparison and statistics, if a mask image is given with the expected results
	Mat results = ffn->EvaluateWithComparison(rulefile, TH, testImage, testMask, statsfile, outImage, col);

	//evaluate simply
	Mat results2 = ffn->filterWithRBF(rulefile, testimg, TH, col);
	
	imshow("results", results2);
	cv::cvtColor(testimg, testimg, COLOR_Lab2BGR); //swap this to the desired color space
	imshow("test img", testimg);
	cout << "\n Job's done!";
	waitKey(0);



	return 1;
}