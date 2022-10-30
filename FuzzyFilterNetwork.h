#pragma once
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;
# define M_PI           3.14159265358979323846  /* pi */
enum mode { DO_IT_ALL, MAKE_TRAINING_DATA, TRAIN_RBF, TEST_RBF };
enum colorspace { RGB, HSV, LAB, LUV };
class FuzzyFilterNetwork
{
private:
		
	int N = 3; //the dimension of the problem, 3 for color filtering

	struct Data
	{
		float** training;
		float** testing;
		int trainingQuanity;
		int testingQuanity;
	};

	struct findings {
		float TP;
		float FP;
		float TN;
		float FN;
	};
public:
	FuzzyFilterNetwork(int n)
	{
		N = n;
	}

	void getDataFromPic(Mat image, Mat mask, int*& dataOut, int*& targetout)
	{
		int N = 3;
		int NoP = image.size[0] * image.size[1];
		cout << "\n NoP=" << NoP;
		dataOut = new int[NoP * 3];
		targetout = new int[NoP];
		for (int i = 0; i < NoP; i++)
		{
			for (int j = 0; j < N; j++)
			{
				dataOut[i * 3 + j] = 0;
			}
			targetout[i] = 0;
		}

		int p = 0;
		short x, y, z; 	char ci; char ct;
		for (int i = 0; i < image.size[0]; i++)
		{
			for (int j = 0; j < image.size[1]; j++)
			{
				//get its color
				x = image.at<Vec3b>(i, j)[0];
				y = image.at<Vec3b>(i, j)[1];
				z = image.at<Vec3b>(i, j)[2];

				dataOut[p * 3 + 0] = x;
				dataOut[p * 3 + 1] = y;
				dataOut[p * 3 + 2] = z;

				if (mask.at<Vec3b>(i, j)[2] >= 100)
				{
					if (mask.at<Vec3b>(i, j)[0] == 0) targetout[p] = 2;
					else
					{
						if (mask.at<Vec3b>(i, j)[0] == 30) targetout[p] = 1;
						else
						{
							if (mask.at<Vec3b>(i, j)[0] == 150) targetout[p] = 3;
						}
					}
				}
				else targetout[p] = 0;
				p++;
			}
		}
	}

	// - dataIn: NxP input data
	// - targetIn: 1xP input classes
	// - dataOut: NxP2 input data, NULL
	// - targetOut: 1xP2 input classes, NULL
	// - returns with the number of output samples
	int cropRedundancyAndInconsistencyFast(int P, int N, int* dataIn, int* targetIn, int*& dataOut, int*& targetout)
	{
		if (!dataIn || !targetIn)
		{
			return 0;
		}

		int* tempDataIdx = new int[P]; //indices for dataIn
		char* zeroFlags = new char[P]; //indices for dataIn
		for (int p = 1; p < P; p++) tempDataIdx[p] = 0;
		for (int p = 0; p < P; p++) zeroFlags[p] = 0;

		tempDataIdx[0] = 0; //the first sample will surely be on the temp list
		int NoP = 1;
		int alreadyHas = 0;
		int parts = 0;

		for (int p = 1; p < P; p++)
		{
			alreadyHas = 0;
			for (int i = 0; i < NoP && !alreadyHas; i++)
			{
				parts = 0;
				for (int j = 0; j < N; j++)
				{
					if (dataIn[j + N * tempDataIdx[i]] == dataIn[j + N * p])
					{
						parts++;
					}
				}
				if (parts == N)
				{
					alreadyHas = 1;
					if (targetIn[p] == 0 || targetIn[p] != targetIn[tempDataIdx[i]]) zeroFlags[i] = 1;
				}

			}
			if (!alreadyHas)
			{
				tempDataIdx[NoP] = p;
				NoP++;
			}
		}

		dataOut = new int[NoP * 3];
		targetout = new int[NoP];
		for (int i = 0; i < NoP; i++)
		{
			for (int j = 0; j < N; j++)
			{
				dataOut[i * 3 + j] = dataIn[tempDataIdx[i] * 3 + j];
			}
			if (zeroFlags[i]) targetout[i] = 0; else targetout[i] = targetIn[tempDataIdx[i]];
		}
		delete[] tempDataIdx;
		delete[] zeroFlags;
		return NoP;
	}

	int ClusteringColors(string dataFile, string ruleFile, double limit)
	{
		int N = 3;

		int P;
		fstream myfile;
		myfile.open(dataFile, ios::in); 	
		myfile >> P;
		int* data = new int[N * P]; 
		int* target = new int[P]; 
		double* minRadius = new double[P];


		char* zeroFlags = new char[P];
		for (int p = 0; p < P; p++) zeroFlags[p] = 0;
		//cout << "\n Received P:" << P;
		for (int p = 0; p < P; p++)
		{
			for (int i = 0; i < N; i++) myfile >> data[p * N + i];
			myfile >> target[p];
		}
		myfile.close();

		//for (int i = P - 30; i < P; i++)
		//{
		//	cout << "\n";
		//	for (int j = 0; j < 3; j++)
		//	{
		//		cout << " " << data[j + i * 3];
		//	}
		//	cout << " " << target[i];
		//}


		clock_t begin, end;
		double elapsed_secs;

		begin = clock();

		//----------------------------------------------------------------------------------------------
		// calculating the smallest distances to other class samples
		//----------------------------------------------------------------------------------------------
		double minRad;
		double dist;
		for (int p = 0; p < P; p++)
		{
			minRad = 1000;
			for (int i = 0; i < P; i++)
			{
				if (i != p)
				{
					//if their classes are not the same
					if (target[p] != target[i])
					{
						dist = ((double)data[i * 3 + 0] - (double)data[p * 3 + 0]) * ((double)data[i * 3 + 0] - (double)data[p * 3 + 0]);
						dist += ((double)data[i * 3 + 1] - (double)data[p * 3 + 1]) * ((double)data[i * 3 + 1] - (double)data[p * 3 + 1]);
						dist += ((double)data[i * 3 + 2] - (double)data[p * 3 + 2]) * ((double)data[i * 3 + 2] - (double)data[p * 3 + 2]);
						dist = sqrt(dist);
						if (dist < minRad) minRad = dist;
					}
				}
			}
			minRadius[p] = minRad / 2;
			if (minRadius[p] > limit)
			{
				minRadius[p] = limit;
			}
		}

		end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; cout << "\nTime spent on distance calculation:" << elapsed_secs;

		//----------------------------------------------------------------------------------------------
		// calculating the relative coverages, flagging covered ones
		//----------------------------------------------------------------------------------------------
		begin = clock();
		for (int p = 0; p < P; p++)
		{
			for (int i = 0; i < P; i++)
			{
				if (i != p)
				{
					//if their classes are the same
					if (target[p] == target[i] && minRadius[p] > minRadius[i])
					{
						dist = ((double)data[i * 3 + 0] - (double)data[p * 3 + 0]) * ((double)data[i * 3 + 0] - (double)data[p * 3 + 0]);
						dist += ((double)data[i * 3 + 1] - (double)data[p * 3 + 1]) * ((double)data[i * 3 + 1] - (double)data[p * 3 + 1]);
						dist += ((double)data[i * 3 + 2] - (double)data[p * 3 + 2]) * ((double)data[i * 3 + 2] - (double)data[p * 3 + 2]);
						dist = sqrt(dist);
						if (zeroFlags[p] - dist > minRadius[i] * 0.6) zeroFlags[i];
					}
				}
			}
		}
		end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; cout << "\nTime spent on coverage calculation:" << elapsed_secs;

		int nonFlagged = 0;
		int nonZero = 0;
		int nonZeroNonFlagged = 0;

		for (int p = 0; p < P; p++)
		{
			if (zeroFlags[p] == 0) nonFlagged++;
			if (target[p] != 0) nonZero++;
			if (zeroFlags[p] == 0 && target[p] != 0)
			{
				nonZeroNonFlagged++;
			}
		}

		begin = clock();
		for (int i = 0; i < 10; i++)
			for (int p = 0; p < P; p++)
			{
				if (zeroFlags[p] == 0 && target[p] != 0)
				{
					double ddd = ((-1) * minRadius[p] * minRadius[p]) / log(0.5);
					//cout << "\n" << log(0.5);
				}
			}
		end = clock(); elapsed_secs = double(end - begin) / (CLOCKS_PER_SEC / 1000); cout << "\nTime spent on sigma calculation: (ms)" << elapsed_secs;

		fstream myfile2;
		myfile2.open(ruleFile, ios::out); //"pos\\WL\\pos.txt"	
		myfile2 << nonZeroNonFlagged << "\n";
		double minminRad;
		for (int p = 0; p < P; p++)
		{
			if (zeroFlags[p] == 0 && target[p] != 0)
			{
				minminRad = minRadius[p];
				if (minminRad > limit) minminRad = limit;
				//double ddd = ((-1)*minminRad * minminRad) / log(0.5);
				double ddd = sqrt(((-1) * minminRad * minminRad) / (2 * log(0.5))); //<---------------------------------------here
				myfile2 << data[p * 3 + 0] << " " << data[p * 3 + 1] << " " << data[p * 3 + 2] << " " << ddd << " " << target[p] << "\n";
			}
		}
		myfile2.close();

		cout << "\n nonflagged:" << nonFlagged;
		cout << "\n nonZero:" << nonZero;
		cout << "\n nonZeroNonFlagged:" << nonZeroNonFlagged << "\n";
		delete[] data;
		delete[] target;
		delete[] minRadius;
		delete[] zeroFlags;
		return 0;
	};

	double gauss_distribution3D(int x1, int x2, int y1, int y2, int z1, int z2, double s)
	{		
		double p1 = ((double)x1 - (double)x2);
		p1 *= p1;
		double p2 = ((double)y1 - (double)y2);
		p2 *= p2;
		double p3 = ((double)z1 - (double)z2);
		p3 *= p3;

		p1 = (p1 + p2 + p3) / ((-2) * s * s);
		return exp(p1);
	}
	/// <summary>
	/// Evaluates the FFN classifier on a given image, given the rules and theta parameter. 
	/// </summary>
	/// <param name="ruleFile">: The text file with the rules</param>
	/// <param name="theta">: threshold; i.e. how close a color tone needs to be to any rule to be considered at all</param>
	/// <param name="col">: the desired color space</param>
	/// <returns></returns>
	Mat filterWithRBF(string rulefile, Mat image, double THRESH, colorspace col)
	{
		int maxC = 0;
		double maxV = 0;
		int NoR;
		switch (col)
		{
		case RGB:
			break;
		case HSV:
			cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
			break;
		case LAB:
			cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
			break;
		case LUV:
			cv::cvtColor(image, image, cv::COLOR_BGR2Luv);
			break;
		}
		fstream myfile;
		myfile.open(rulefile, ios::in); 
		myfile >> NoR;

		int* data = new int[3 * NoR];
		int* target = new int[NoR]; 
		double* sigmas = new double[NoR];
		double tmp;
		for (int p = 0; p < NoR; p++)
		{
			myfile >> tmp;
			data[p * 3 + 0] = tmp;
			myfile >> tmp;
			data[p * 3 + 1] = tmp;
			myfile >> tmp;
			data[p * 3 + 2] = tmp;
			myfile >> sigmas[p];
			myfile >> tmp;
			target[p] = tmp;
			//cout << "\n" << p << " - " << data[p * 3 + 0] << " , " << data[p * 3 + 1] << " , " << data[p * 3 + 2] << " | " << sigmas[p] << " | " << target[p];
		}
		myfile.close();

		int closestdata1, closestdata2, closestdata3;
		Mat result;
		result.create(image.size(), image.type());
		result = Scalar::all(0);
		cv::cvtColor(result, result, COLOR_BGR2HSV);

		int x, y, z, c1, c2, c3, t; double s;

		for (int i = 1; i < image.size.p[0]; i++)
		{
			for (int j = 1; j < image.size.p[1]; j++)
			{
				x = image.at<Vec3b>(i, j)[0];
				y = image.at<Vec3b>(i, j)[1];
				z = image.at<Vec3b>(i, j)[2];
				maxV = 0.01;
				maxC = -1;
				for (int c = 0; c < NoR; c++)
				{
					c1 = data[c * 3 + 0];
					c2 = data[c * 3 + 1];
					c3 = data[c * 3 + 2];
					s = sigmas[c];
					t = target[c];

					double g = gauss_distribution3D(x, c1, y, c2, z, c3, s);
					if (g > maxV)
					{
						maxV = g;
						maxC = t;
					}

				}
				if (maxV > THRESH)
				{
					//cout << " WINNER";
					result.at<Vec3b>(i, j)[1] = 255;
					result.at<Vec3b>(i, j)[2] = 255;

					switch (maxC)
					{
					case 1:result.at<Vec3b>(i, j)[0] = 30; break;
					case 2:result.at<Vec3b>(i, j)[0] = 0; break;
					case 3:result.at<Vec3b>(i, j)[0] = 150; break;
					}
				}
			}
		}
		cv::cvtColor(result, result, COLOR_HSV2BGR);
		//imshow("results", result);
		//imshow("input", image);
		//waitKey(0);

		delete[] data;
		delete[] target;
		delete[] sigmas;
		return result;
	}


/*	Mat filterWithRBF(char* rulefile, Mat image, double THRESH)
	{
		int maxC = 0;
		double maxV = 0;
		int NoR;

		fstream myfile;
		myfile.open(rulefile, ios::in); //"pos\\WL\\pos.txt"	
		myfile >> NoR;

		int* data = new int[3 * NoR]; //indices for dataIn
		int* target = new int[NoR]; //
		double* sigmas = new double[NoR];

		for (int p = 0; p < NoR; p++)
		{
			myfile >> data[p * 3 + 0];
			myfile >> data[p * 3 + 1];
			myfile >> data[p * 3 + 2];
			myfile >> sigmas[p];
			myfile >> target[p];
		}
		myfile.close();

		//cout << "NoR=" << NoR;

		//for (int i = 0; i < 10; i++)
		//{
		//	cout << "\n";
		//	for (int j = 0; j < 3; j++)
		//	{
		//		cout << " " << data[j + i * 3];
		//	}
		//	cout << " " << sigmas[i];
		//	cout << " " << target[i];
		//}
		int closestdata1, closestdata2, closestdata3;
		Mat result;
		result.create(image.size(), image.type());
		result = Scalar::all(0);
		cv::cvtColor(result, result, COLOR_BGR2HSV);

		int x, y, z, c1, c2, c3, t; double s;

		for (int i = 1; i < image.size.p[0]; i++)
			//for (int i = 1; i < 10; i++)
		{
			for (int j = 1; j < image.size.p[1]; j++)
				//for (int j = 1; j < 10; j++)
			{
				x = image.at<Vec3b>(i, j)[0];
				y = image.at<Vec3b>(i, j)[1];
				z = image.at<Vec3b>(i, j)[2];
				maxV = 0.1;
				maxC = -1;
				for (int c = 0; c < NoR; c++)
				{
					c1 = data[c * 3 + 0];
					c2 = data[c * 3 + 1];
					c3 = data[c * 3 + 2];
					s = sigmas[c];
					t = target[c];

					//if (i == 9 && j == 9 && c1 == 76 && c2 == 80 && c3 == 159)
					//{
					//	cout << "\n s=" << s;
					//	double p1 = (x - c1); cout << "\n " << p1;
					//	p1 *= p1; cout << "\n " << p1;
					//	double p2 = (y - c2); cout << "\n " << p2;
					//	p2 *= p2; cout << "\n " << p2;
					//	double p3 = (z - c3); cout << "\n " << p3;
					//	p3 *= p3; cout << "\n " << p3;

					//	p1 = (p1 + p2 + p3) / ((-2)*s*s); cout << "\n " << p1;
					//	p1 = exp(p1); cout << "\n " << p1;
					//}

					double g = gauss_distribution3D(x, c1, y, c2, z, c3, s);

					if (g > maxV)
					{
						maxV = g;
						maxC = t;
						//closestdata1 = c1;
						//closestdata2 = c2;
						//closestdata3 = c3;
					}

				}
				if (maxV > THRESH)
				{
					result.at<Vec3b>(i, j)[1] = 255;
					result.at<Vec3b>(i, j)[2] = 255;

					switch (maxC)
					{
					case 1:result.at<Vec3b>(i, j)[0] = 30; break;
					case 2:result.at<Vec3b>(i, j)[0] = 0; break;
					case 3:result.at<Vec3b>(i, j)[0] = 150; break;
					}
				}
				//if (i == 9 && j == 9)
				//{
				//	cout << "\n" << maxV << " " << maxC;
				//	cout << " | " << c1 << " " << c2 << " " << c3;
				//	cout << " | " << closestdata1 << " "<< closestdata2<< " " << closestdata3;
				//}
			}
		}
		cv::cvtColor(result, result, COLOR_HSV2BGR);
		delete[] data;
		delete[] target;
		delete[] sigmas;
		return result;
	}
	*/


	findings compareImageToEtalon(cv::Mat img, cv::Mat etalon)
	{
		findings F0;
		F0.TP = 0;
		F0.FP = 0;
		F0.TN = 0;
		F0.FN = 0;

		for (int j = 0; j < img.cols; j++)
			for (int i = 0; i < img.rows; i++)
			{
				if (img.at<Vec3b>(i, j)[2] == 0) // if testimage says N
				{
					if (etalon.at<Vec3b>(i, j)[2] == 0) // if etalon also says N
					{
						F0.TN++;
					}
					else
					{
						F0.FN++;
					}
				}
				else //if testimage says P
				{
					if (etalon.at<Vec3b>(i, j)[2] == 0) //if etalon says N
					{
						F0.FP++;
					}
					else
					{
						if (etalon.at<Vec3b>(i, j)[0] == img.at<Vec3b>(i, j)[0] && etalon.at<Vec3b>(i, j)[1] == img.at<Vec3b>(i, j)[1])
						{ //if they have the same color
							F0.TP++;
						}
						else F0.FP++;
					}
				}
			}
		return F0;
	}

	//if you use a custom clustering algorithm (e.g. Mean-shift or K-means), then the rules can be made with the following method
	int MakeRulesFromOtherClustering(string dataFile0, string dataFile1, string dataFile2, string dataFile3, string ruleFile, double limit)
	{
		int N = 3;
		//int * dataIn, int * targetIn;
		//int * data = new int[N*P]; //indices for dataIn


		int P0, P1, P2, P3;
		fstream myfile0, myfile1, myfile2, myfile3;
		myfile0.open(dataFile0, ios::in); //this is the long list of negative datapoints
		myfile1.open(dataFile1, ios::in); //class 1, P1
		myfile2.open(dataFile2, ios::in);
		myfile3.open(dataFile3, ios::in);
		myfile0 >> P0; cout << "\nP0=" << P0;
		myfile1 >> P1; cout << "\nP1=" << P1;
		myfile2 >> P2; cout << "\nP2=" << P2;
		myfile3 >> P3; cout << "\nP3=" << P3;
		int P = P0 + P1 + P2 + P3; cout << "\nP=" << P;
		int kezdohelyek[5];
		kezdohelyek[0] = 0; kezdohelyek[1] = P0; kezdohelyek[2] = P0 + P1; kezdohelyek[3] = P0 + P1 + P2; kezdohelyek[4] = P;

		double* data = new double[N * P]; //indices for dataIn
		int* target = new int[P]; //
		double* minRadius = new double[P];


		//	char * zeroFlags = new char[P];
		//	for (int p = 0; p < P; p++) zeroFlags[p] = 0;
		cout << "\n Received P:" << P;
		cout << "\n reading from " << kezdohelyek[0] << " to " << kezdohelyek[1];
		for (int p = kezdohelyek[0]; p < kezdohelyek[1]; p++)
		{
			//cout << p << ": ";
			for (int i = 0; i < N; i++)
			{
				myfile0 >> data[p * N + i];
				//cout << data[p*N + i] << " ";
			}
			myfile0 >> target[p];
			//cout << target[p] << "\n";
		}
		myfile0.close();
		cout << "\n reading from " << kezdohelyek[1] << " to " << kezdohelyek[2];
		for (int p = kezdohelyek[1]; p < kezdohelyek[2]; p++)
		{
			for (int i = 0; i < N; i++) myfile1 >> data[p * N + i];
			myfile1 >> target[p];
		}
		myfile1.close();
		cout << "\n reading from " << kezdohelyek[2] << " to " << kezdohelyek[3];
		for (int p = kezdohelyek[2]; p < kezdohelyek[3]; p++)
		{
			for (int i = 0; i < N; i++) myfile2 >> data[p * N + i];
			myfile2 >> target[p];
		}
		myfile2.close();
		cout << "\n reading from " << kezdohelyek[3] << " to " << kezdohelyek[4];
		for (int p = kezdohelyek[3]; p < kezdohelyek[4]; p++)
		{
			for (int i = 0; i < N; i++) myfile3 >> data[p * N + i];
			myfile3 >> target[p];
		}
		myfile3.close();


		for (int i = P - 30; i < P; i++)
		{
			cout << "\n " << i << ": ";
			for (int j = 0; j < 3; j++)
			{
				cout << " " << data[j + i * 3];
			}
			cout << " " << target[i];
		}


		clock_t begin, end;
		double elapsed_secs;

		begin = clock();

		double minRad;
		double dist;
		for (int k = 1; k < 4; k++)
		{
			for (int p = kezdohelyek[k]; p < kezdohelyek[k + 1]; p++)
			{
				minRad = 1000;
				for (int i = 0; i < kezdohelyek[k]; i++)
				{
					dist = ((double)data[i * 3 + 0] - (double)data[p * 3 + 0]) * ((double)data[i * 3 + 0] - (double)data[p * 3 + 0]);
					dist += ((double)data[i * 3 + 1] - (double)data[p * 3 + 1]) * ((double)data[i * 3 + 1] - (double)data[p * 3 + 1]);
					dist += ((double)data[i * 3 + 2] - (double)data[p * 3 + 2]) * ((double)data[i * 3 + 2] - (double)data[p * 3 + 2]);
					dist = sqrt(dist);
					if (dist < minRad) minRad = dist;
				}
				for (int i = kezdohelyek[k + 1]; i < P; i++)
				{
					dist = ((double)data[i * 3 + 0] - (double)data[p * 3 + 0]) * ((double)data[i * 3 + 0] - (double)data[p * 3 + 0]);
					dist += ((double)data[i * 3 + 1] - (double)data[p * 3 + 1]) * ((double)data[i * 3 + 1] - (double)data[p * 3 + 1]);
					dist += ((double)data[i * 3 + 2] - (double)data[p * 3 + 2]) * ((double)data[i * 3 + 2] - (double)data[p * 3 + 2]);
					dist = sqrt(dist);
					if (dist < minRad) minRad = dist;
				}
				minRadius[p] = minRad / 2;
				if (minRadius[p] > limit)
				{
					minRadius[p] = limit;
				}
			}
		}

		fstream ruleFILE;
		ruleFILE.open(ruleFile, ios::out); //"pos\\WL\\pos.txt"	
	//	myfile2 << nonZeroNonFlagged << "\n";
		ruleFILE << (P1 + P2 + P3) << "\n";
		double minminRad;
		for (int p = kezdohelyek[1]; p < P; p++)
		{
			minminRad = minRadius[p];
			if (minminRad > limit) minminRad = limit;
			//double ddd = ((-1)*minminRad * minminRad) / log(0.5);
			double ddd = sqrt(((-1) * minminRad * minminRad) / (2 * log(0.5))); //<---------------------------------------here
			ruleFILE << data[p * 3 + 0] << " " << data[p * 3 + 1] << " " << data[p * 3 + 2] << " " << ddd << " " << target[p] << "\n";
		}
		ruleFILE.close();


		end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; cout << "\nTime spent on distance calculation:" << elapsed_secs;
		delete[] data;
		delete[] target;
		delete[] minRadius;
		
		return 0;
	};

	void comparison(Mat img, Mat etalon, string filename)
	{
		cv::cvtColor(img, img, COLOR_BGR2HSV);
		cv::cvtColor(etalon, etalon, COLOR_BGR2HSV);

		findings F0;
		F0.TP = 0;
		F0.FP = 0;
		F0.TN = 0;
		F0.FN = 0;

		F0 = compareImageToEtalon(img, etalon);

		ofstream myfile;
		myfile.open(filename);
		cout << "\n TP = " << F0.TP; myfile << F0.TP;
		cout << "\n FP = " << F0.FP; myfile << "\n" << F0.FP;
		cout << "\n TN = " << F0.TN; myfile << "\n" << F0.TN;
		cout << "\n FN = " << F0.FN; myfile << "\n" << F0.FN;
		myfile << "\n";
		cout << "\n TPR = " << F0.TP / (F0.TP + F0.FN) * 100 << " (recall)"; float TPR = F0.TP / (F0.TP + F0.FN); myfile << "\n" << TPR * 100;
		cout << "\n PPV = " << F0.TP / (F0.TP + F0.FP) * 100 << " (precision)"; myfile << "\n" << F0.TP / (F0.TP + F0.FP) * 100;
		cout << "\n TNR = " << F0.TN / (F0.TN + F0.FP) * 100 << " (specificity)"; float TNR = F0.TN / (F0.TN + F0.FP);	myfile << "\n" << TNR * 100;
		cout << "\n FNR = " << F0.FN / (F0.FN + F0.TP) * 100 << " (miss rate)";  myfile << "\n" << F0.FN / (F0.FN + F0.TP) * 100;
		cout << "\n FPR = " << F0.FP / (F0.FP + F0.TN) * 100 << " (false alarm rate)";  myfile << "\n" << F0.FP / (F0.FP + F0.TN) * 100;
		cout << "\n ACC = " << (F0.TP + F0.TN) / (F0.TP + F0.FN + F0.FP + F0.TN) * 100 << " (Accuracy)";  myfile << "\n" << (F0.TP + F0.TN) / (F0.TP + F0.FN + F0.FP + F0.TN) * 100;
		cout << "\n BA  = " << (TPR + TNR) / (2) * 100 << " (Balanced Accuracy)";  myfile << "\n" << (TPR + TNR) / (2) * 100;

		myfile.close();
	}

	void Launch(mode currentMode, colorspace col)
	{

	}
	/// <summary>
	/// Creates clusters based on the training data, using limit as the clustering parameter
	/// </summary>
	/// <param name="trainingData">: the training data file</param>
	/// <param name="rulefile">: the resulting rules</param>
	/// <param name="limit">: the parameter for the clustering</param>
	void TrainFFN(string trainingData, string rulefile, double limit)
	{
		clock_t begin, end;
		double elapsed_secs;

		begin = clock();
		ClusteringColors(trainingData, rulefile, limit);
		end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; cout << "\nTime spent on getDataFromPic:" << elapsed_secs;

	}
	void MakeTrainingData(string imageFile, string maskFile, string trainingDataFile, colorspace col)
	{
		/// Load an image
		Mat trainimgHSV, trainmaskHSV;
		Mat trainimg = imread(imageFile.c_str(), cv::IMREAD_COLOR);
		Mat trainmask = imread(maskFile.c_str(), cv::IMREAD_COLOR);
		switch (col)
		{
		case RGB:
			break;
		case HSV:
			cv::cvtColor(trainimg, trainimg, cv::COLOR_BGR2HSV);
			break;
		case LAB:
			cv::cvtColor(trainimg, trainimg, cv::COLOR_BGR2Lab);
			break;
		case LUV:
			cv::cvtColor(trainimg, trainimg, cv::COLOR_BGR2Luv);
			break;
		}

		cvtColor(trainmask, trainmaskHSV, cv::COLOR_BGR2HSV);

		int size = trainimg.size[0] * trainimg.size[1];
		int* trainingData1 = NULL;
		int* trainingTargets1 = NULL;

		clock_t begin, end;
		double elapsed_secs;
		begin = clock();
		getDataFromPic(trainimg, trainmaskHSV, trainingData1, trainingTargets1);
		end = clock(); elapsed_secs = double(end - begin) / (CLOCKS_PER_SEC / 1000); cout << "\nTime spent on getDataFromPic (ms):" << elapsed_secs;

		int* dataOut = NULL;
		int* targetOut = NULL;

		int finalLength;
		begin = clock();
		finalLength = cropRedundancyAndInconsistencyFast(size, 3, trainingData1, trainingTargets1, dataOut, targetOut);
		end = clock(); elapsed_secs = double(end - begin) / (CLOCKS_PER_SEC / 1000); cout << "\nTime spent on cropRedundancyAndInconsistencyFast (ms):" << elapsed_secs;

		fstream f2(trainingDataFile.c_str(), ios::out);
		if (!f2) {
			cout << "Cannot open file.\n";
			waitKey(0);
		}
		else
		{
			f2 << finalLength << "\n";
			for (int i = 0; i < finalLength; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					f2 << dataOut[j + i * 3] << " ";
				}
				f2 << targetOut[i];
				f2 << "\n";
			}
			f2.close();
		}

		if (trainingData1) delete[] trainingData1;
		if (trainingTargets1) delete[] trainingTargets1;
		cout << "\n" << finalLength;

		if (dataOut) delete[] dataOut;
		if (targetOut) delete[] targetOut;
	}
	/// <summary>
	/// Evaluates the FFN classifier on a given image, given the rules and theta parameter. 
	/// It also provides statistics about the accuracy metrics given the mask of the expected classification output.
	/// </summary>
	/// <param name="ruleFile">: The text file with the rules</param>
	/// <param name="theta">: threshold; i.e. how close a color tone needs to be to any rule to be considered at all</param>
	/// <param name="imageFile">: the image that needs to be evaluated</param>
	/// <param name="testMaskFile">: the expected results</param>
	/// <param name="statsFile">: statistics on the classification accuracy</param>
	/// <param name="outImage">: saving the results</param>
	/// <param name="col">: the desired color space</param>
	/// <returns></returns>
	Mat EvaluateWithComparison(string ruleFile, double theta, string imageFile, string testMaskFile, string statsFile, string outImage, colorspace col)
	{
		Mat results;
		Mat testimg = imread(imageFile, cv::IMREAD_COLOR); if (testimg.empty()) { cout << "\n no testimg!"; return results; }
		Mat testmask = imread(testMaskFile, cv::IMREAD_COLOR); if (testmask.empty()) { cout << "\n no testmask!"; return results; }

		clock_t begin, end;
		double elapsed_secs;
		begin = clock();
		results = filterWithRBF(ruleFile, testimg, theta, col);
		end = clock(); elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; cout << "\nTime spent on filterWithRBF:" << elapsed_secs;

		imwrite(outImage, results);

		comparison(results, testmask, statsFile);
		return results;
	}
	~FuzzyFilterNetwork()
	{
	}
};
