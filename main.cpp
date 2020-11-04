#include <opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main_()
{
    VideoCapture* cap = new VideoCapture("/home/pmj-nano/Desktop/1103/130 (2).avi");
    if(!cap->isOpened())
    {
        cout <<"video is empty!!"<<endl;
    }
    Mat frame; //1080 1920
    //discard the first 25 frames
    while(cap->read(frame))
    {
        static unsigned int counter=0;
        if(++counter == 25)
        {
            break;
        }
    }

    Mat meanFrame,stdDevFrame,singleChannel/*IM_bri*/;
    double mean;
    vector<Mat> splitFrame2bgr3;

    Mat meanBri,stdDevBri;
    double thresholdRate;
    //        H_S_f_A_T
    Mat HSFAT = Mat::zeros(85,85,CV_64FC1);
    //
    while(cap->read(frame))
    {
        double time = getTickCount();


        meanStdDev(frame,meanFrame,stdDevFrame);  
        mean /*results_de_mean*/= (meanFrame.at<Vec3d>(0,0)[0] + meanFrame.at<Vec3d>(0,0)[1] + meanFrame.at<Vec3d>(0,0)[2])/3;
//        cout <<"mean:"<<mean<<endl;
        split(frame,splitFrame2bgr3);
        //
        singleChannel =  0.257*splitFrame2bgr3[2] + 0.564*splitFrame2bgr3[1] + 0.098*splitFrame2bgr3[0] +0.0;
        
//        IM_bri_T_mean = mean(IM_bri(:));
//        IM_bri_T_var = var(IM_bri(:));
        meanStdDev(singleChannel,meanBri,stdDevBri);
//          threshold_rate = IM_bri_T_mean/IM_bri_T_var*IM_bri_T_mean/80;
        thresholdRate = meanBri.at<double>(0,0)/stdDevBri.at<double>(0,0)*meanBri.at<double>(0,0)/80;
//        threshold_rate = threshold_rate + 0.2;
        thresholdRate +=0.2;
        
//        if threshold_rate > 0.6
//            threshold_rate = 0.6;
//        end
        if(thresholdRate > 0.6)
        {
            thresholdRate = 0.6;
        }
        
//        H_S_f_A_T = H_S_f_A_arr0(:,:,fix(threshold_rate*10));%根据threshold_rate，获取卷积核
        
        
        imshow("frame",frame);
        waitKey(10);
        double fps = getTickFrequency()/(getTickCount() - time);
        cout <<"fps"<<fps<<endl;
    }

    return 0;
}
