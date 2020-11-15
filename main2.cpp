#include <opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;


int mainfilter()
{
//    Mat src = Mat_<unsigned int>(2,2)<<[1,2,3,4];
    float a[] = {1,2,3,4};
    float b[] = {-1,1,-2,2};
    Mat aImage = Mat(2,2,CV_32FC1,&a);
    Mat bImage = Mat(2,2,CV_32FC1,&b);

//    copyMakeBorder(aImage,aImage,1,1,1,1,BORDER_CONSTANT,Scalar(0));
    filter2D(aImage,aImage,-1,bImage,Point(-1,-1),0,BORDER_CONSTANT);
    for(int i=0;i<aImage.rows;i++)
    {
        for(int j=0;j<aImage.cols;j++)
        {
           cout <<aImage.at<float>(i,j)<<" ";
        }
        cout <<endl;
    }
    printf("%lf\n",pow(2,3));

    return 0;
}
