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
Vec3b getColorSubpix(const Mat& img, Point2f pt)
{
    Mat patch;
    getRectSubPix(img, Size(1,1), pt, patch);
    return patch.at<Vec3b>(0,0);
}*/

//#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pybind11 { namespace detail{
template<>
struct type_caster<Mat>{
public:
    PYBIND11_TYPE_CASTER(Mat, _("numpy.ndarray"));

    //! 1. cast numpy.ndarray to Mat
    bool load(handle obj, bool){
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        //const int ndims = (int)info.ndim;
        int nh = 1;
        int nw = 1;
        int nc = 1;
        int ndims = info.ndim;
        if(ndims == 2){
            nh = info.shape[0];
            nw = info.shape[1];
        } else if(ndims == 3){
            nh = info.shape[0];
            nw = info.shape[1];
            nc = info.shape[2];
        }else{
            char msg[64];
            sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
            throw logic_error(msg);
            return false;
        }

        int dtype;
        if(info.format == format_descriptor<unsigned char>::format()){
            dtype = CV_8UC(nc);
        }else if (info.format == format_descriptor<int>::format()){
            dtype = CV_32SC(nc);
        }else if (info.format == format_descriptor<float>::format()){
            dtype = CV_32FC(nc);
        }else{
            throw logic_error("Unsupported type, only support uchar, int32, float");
            return false;
        }

        value = Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    //! 2. cast Mat to numpy.ndarray
    static handle cast(const Mat& mat, return_value_policy, handle defval){
        CV_UNUSED(defval);

        string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type)? 2 : 3;

        if(depth == CV_8U){
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }else if(depth == CV_32S){
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }else if(depth == CV_32F){
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }else{
            throw logic_error("Unsupport type, only support uchar, int32, float");
        }

        vector<size_t> bufferdim;
        vector<size_t> strides;
        if (dim == 2) {
            bufferdim = {(size_t) nh, (size_t) nw};
            strides = {elemsize * (size_t) nw, elemsize};
        } else if (dim == 3) {
            bufferdim = {(size_t) nh, (size_t) nw, (size_t) nc};
            strides = {(size_t) elemsize * nw * nc, (size_t) elemsize * nc, (size_t) elemsize};
        }
        return array(buffer_info( mat.data,  elemsize,  format, dim, bufferdim, strides )).release();
    }
};
}}//! end namespace pybind11::detail

int add23(int i, int j) {
    return i + j;
}


bool cmp(KeyPoint kp1, KeyPoint kp2)
{
    return kp1.response > kp2.response;
}


vector<float> sphorb(string image,int keypoint)
{

	SPHORB sorb(keypoint);

    Mat img1 = imread(image);

	Mat descriptors1;

    vector<KeyPoint> kPoint1;

	sorb(img1, Mat(), kPoint1, descriptors1);
    int tam = kPoint1.size();
    //const int height = tam;
    //const int widht = 3;


    //unsigned char key1[height][widht];
    vector<float>key1; 
    for(int i=0; i<tam; i++)
    {
        KeyPoint kp1 = kPoint1[i];
        key1.push_back(float(kp1.pt.x));
        key1.push_back(float(kp1.pt.y));
        key1.push_back(1.0f);
        //key1[i][0] = float(kp1.pt.x);
        //key1[i][1] = float(kp1.pt.y);
        //key1[i][2] = float(1.0f);
    }
    
    float delimiter = int(key1.size());

    for(int i=0; i<descriptors1.rows; i++)
    {
        int tam = descriptors1.cols;
        for(int j=0; j<tam; j++)
            key1.push_back(float((int) descriptors1.at<uchar>(i,j)));
    }
    key1.push_back(delimiter);


    return key1;
}



/*int main(int argc, char * argv[])
{

    	int maxpoints = 400;
	float ratio = 0.75f;
	SPHORB sorb(12000);

	Mat img1 = imread(argv[1]);
	Mat img2 = imread(argv[2]);
    	string destiny_path = argv[3];

	Mat descriptors1;
	Mat descriptors2;

	vector<KeyPoint> kPoint1;
	vector<KeyPoint> kPoint2;

	sorb(img1, Mat(), kPoint1, descriptors1);
	sorb(img2, Mat(), kPoint2, descriptors2); 

	BFMatcher matcher(NORM_HAMMING, false);
	Matches matches;
	
	vector<Matches> dupMatches;
	matcher.knnMatch(descriptors1, descriptors2, dupMatches, 2);
    ratioTest(dupMatches, ratio, matches);


    //matches.sort(cmp);

    ofstream puntos1, puntos2;
    puntos1.open(destiny_path+"puntos1.dat");

    for(int i=0; i<kPoint1.size();i++)
    {
        KeyPoint kp1 = kPoint1[i];
        puntos1 << kp1.pt.x << " " << kp1.pt.y << " " << 1 << endl;
    }
    puntos1.close();
    
    puntos2.open(destiny_path+"puntos2.dat");

    for(int i=0; i<kPoint2.size();i++)
    {
        KeyPoint kp1 = kPoint2[i];
        puntos2 << kp1.pt.x << " " << kp1.pt.y << " " << 1 << endl;
    }
    puntos2.close();
   
    ofstream desc1, desc2;

    desc1.open(destiny_path+"desc1.dat");
    for(int i=0; i<descriptors1.rows; i++)
    {
        int tam = descriptors1.cols-1;
        for(int j=0; j<tam; j++)
        {
            desc1 << (int) descriptors1.at<uchar>(i,j) << " ";
        }
        desc1 << (int) descriptors1.at<uchar>(i,tam) << endl;
    }
    desc1.close();

    desc2.open(destiny_path+"desc2.dat");
    for(int i=0; i<descriptors2.rows; i++)
    {
        int tam = descriptors2.cols-1;
        for(int j=0; j<tam; j++)
        {
            desc2 << (int) descriptors2.at<uchar>(i,j) << " ";
        }
        desc2 << (int) descriptors2.at<uchar>(i,tam) << endl;
    }
    desc2.close();


	ofstream myfile;
  	myfile.open (destiny_path+"p1.dat");

	for(int i = 0; i < matches.size(); i++)
    {
        KeyPoint kp1 = kPoint1[matches.at(i).queryIdx];
	    KeyPoint kp2 = kPoint2[matches.at(i).trainIdx];
        myfile << kp1.pt.x << " " << kp1.pt.y << " " << 1 << endl;
    }
	myfile.close();


	
	ofstream myfile2;
  	myfile2.open(destiny_path+"p2.dat");

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
  	myfile3.open(destiny_path+"mismatch.dat");
	

	myfile3<<kPoint1.size()<<" "<<kPoint2.size()<<" "<<matches.size()<<endl;

    myfile3.close();
	Mat imgMatches;
	::drawMatches(img1, kPoint1, img2, kPoint2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1),  
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,true);

    imwrite("1_matches.jpg", imgMatches);

	return 0;
}*/

PYBIND11_MODULE(sphorb_cpp, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add23, "A function which adds two numbers");
    //m.def("sphorb_key",&keypoint, "sphorb keypoints");
    //m.def("sphorb_desc",&descriptor, "sphorb descriptors");
    m.def("sphorb",&sphorb, "sphorb descriptors");
    
}


