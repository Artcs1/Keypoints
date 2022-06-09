/*
	AUTHOR:
	Qiang Zhao, email: qiangzhao@tju.edu.cn
	Copyright (C) 2015 Tianjin University
	School of Computer Software
	School of Computer Science and Technology

	LICENSE:
	SPHORB is distributed under the GNU General Public License.  For information on 
	commercial licensing, please contact the authors at the contact address below.

	REFERENCE:
	@article{zhao-SPHORB,
	author   = {Qiang Zhao and Wei Feng and Liang Wan and Jiawan Zhang},
	title    = {SPHORB: A Fast and Robust Binary Feature on the Sphere},
	journal  = {International Journal of Computer Vision},
	year     = {2015},
	volume   = {113},
	number   = {2},
	pages    = {143-159},
	}
*/
#include <fstream>

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "SPHORB.h"
#include "utility.h"
using namespace std;
using namespace cv;

/*
cv::Vec3b getColorSubpix(const cv::Mat& img, cv::Point2f pt)
{
    cv::Mat patch;
    cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
    return patch.at<cv::Vec3b>(0,0);
}*/

bool cmp(KeyPoint kp1, KeyPoint kp2)
{
    return kp1.response > kp2.response;
}

int main(int argc, char * argv[])
{

    int maxpoints = 400;
	float ratio = 0.75f;
	SPHORB sorb;

	Mat img1 = imread(argv[1]);
	Mat img2 = imread(argv[2]);

	Mat descriptors1;
	Mat descriptors2;

	vector<KeyPoint> kPoint1;
	vector<KeyPoint> kPoint2;

	sorb(img1, Mat(), kPoint1, descriptors1);
	sorb(img2, Mat(), kPoint2, descriptors2); 

    printf("%d\n",kPoint1.size());

    for(int i=0; i<kPoint1.size();i++)
        kPoint1[i].class_id = i;
    
    for(int i=0; i<kPoint2.size();i++)
        kPoint2[i].class_id = i;


    sort(kPoint1.begin(),kPoint1.end(),cmp);
    sort(kPoint2.begin(),kPoint2.end(),cmp);



    Mat t1,t2;
    t1 = descriptors1,t2 = descriptors2;

    //Reordering descriptors - kPoint1
    for(int i=0; i<descriptors1.rows; i++)
        t1.row(i) = descriptors1.row(kPoint1[i].class_id);
    descriptors1 = t1;

    //Reorderin descriptros - kPoint2
    for(int i=0; i<descriptors2.rows; i++)
        t2.row(i) = descriptors2.row(kPoint2[i].class_id);
    descriptors2 = t2;


    while(kPoint1.size()>maxpoints)
        kPoint1.pop_back();

    while(kPoint2.size()>maxpoints)
        kPoint2.pop_back();
  

    while(descriptors1.rows>maxpoints)
        descriptors1.pop_back();

    while(descriptors2.rows>maxpoints)
        descriptors2.pop_back();



	BFMatcher matcher(NORM_HAMMING, false);
	Matches matches;
	
	vector<Matches> dupMatches;
	matcher.knnMatch(descriptors1, descriptors2, dupMatches, 2);
	ratioTest(dupMatches, ratio, matches);

    //matches.sort(cmp);

    ofstream puntos1, puntos2;
    puntos1.open("puntos1.dat");

    for(int i=0; i<kPoint1.size();i++)
    {
        KeyPoint kp1 = kPoint1[i];
        puntos1 << kp1.pt.x << " " << kp1.pt.y << " " << 1 << endl;
    }
    puntos1.close();
    
    puntos2.open("puntos2.dat");

    for(int i=0; i<kPoint2.size();i++)
    {
        KeyPoint kp1 = kPoint2[i];
        puntos2 << kp1.pt.x << " " << kp1.pt.y << " " << 1 << endl;
    }
    puntos2.close();
    



	ofstream myfile;
  	myfile.open ("p1.dat");

	for(int i = 0; i < matches.size(); i++)
    {
        KeyPoint kp1 = kPoint1[matches.at(i).queryIdx];
	    KeyPoint kp2 = kPoint2[matches.at(i).trainIdx];
        myfile << kp1.pt.x << " " << kp1.pt.y << " " << 1 << endl;
    }
	myfile.close();


	
	ofstream myfile2;
  	myfile2.open("p2.dat");

	for(int i = 0; i < matches.size(); i++){
	   KeyPoint kp1 = kPoint1[matches.at(i).queryIdx];
	   KeyPoint kp2 = kPoint2[matches.at(i).trainIdx];
           //int b = getColorSubpix(img1, kp1.pt)[0];
	   //int g = getColorSubpix(img1, kp1.pt)[1];
	   //int r = getColorSubpix(img1, kp1.pt)[2];
           myfile2 << kp2.pt.x << " " << kp2.pt.y << " " << 1 << endl;
        }
	myfile2.close();

	ofstream myfile3;
  	myfile3.open("mismatch.dat");
	

	myfile3<<kPoint1.size()<<" "<<kPoint2.size()<<" "<<matches.size()<<endl;

    myfile3.close();
	Mat imgMatches;
	::drawMatches(img1, kPoint1, img2, kPoint2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1),  
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,true);

    imwrite("1_matches.jpg", imgMatches);

	return 0;
}
