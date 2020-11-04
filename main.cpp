#include <opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

vector<Mat> extractConvMat();
void ycbcrUpdate(const Mat& IM_result_cbcr,const Mat& IM_bri_T ,Mat& IM_result_cbcr_re);

int main()
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
    //
    Mat HSFAT = Mat::zeros(85,85,CV_64FC1);
    vector<Mat> hsfatMat = extractConvMat(); //H_S_f_A_T
    Mat IM_bri_T;
    //
    while(cap->read(frame))  //type CV_8UC3
    {
        double time = getTickCount();
        cout <<"frame.type():"<<frame.type()<<endl;
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
        //        IM_bri_T = IM_bri;
        //        H_S_f_A_T = H_S_f_A_arr0(:,:,fix(threshold_rate*10));%根据threshold_rate，获取卷积核
        Mat H_S_f_A_T = hsfatMat[floor(thresholdRate*10)];
        IM_bri_T = singleChannel;


//        IM_bri_T = conv2(IM_bri_T,H_S_f_A_T,'same')*1;%进行卷积运算

        filter2D(IM_bri_T,IM_bri_T,CV_64FC1,H_S_f_A_T);
        Mat meanOfIM_bri_T,stdDevOfIM_bri_T;
        meanStdDev(IM_bri_T,meanOfIM_bri_T,stdDevOfIM_bri_T);
//            IM_bri_T = single(IM_bri_T/mean(IM_bri_T(:))*IM_bri_T_mean/1.5);%调整颜色
        IM_bri_T = IM_bri_T/meanOfIM_bri_T.at<double>(0,0)*meanBri.at<double>(0,0)/1.5;

//        IM_result_cbcr = single(rgb2ycbcr(uint8(IM_result)));%将输入图片rgb转ycbcr
//        IM_result_cbcr(:,:,1) = single(IM_bri_T);%更新y通道的数值
//        IM_result_cbcr_re = ycbcr2rgb(uint8(IM_result_cbcr));%将输出图片ycbcr转rgb

        Mat IM_result_cbcr;
        cvtColor(frame,IM_result_cbcr,COLOR_BGR2YCrCb);
        Mat IM_result_cbcr_re;
        ycbcrUpdate(IM_result_cbcr, IM_bri_T, IM_result_cbcr_re);

//        results_de = single(IM_result_cbcr_re);
//        results_de = uint8(results_de/mean(results_de(:))*results_de_mean);

        imshow("frame",frame);
        imshow("IM_result_cbcr_re",IM_result_cbcr_re);
        waitKey(1);
        double fps = getTickFrequency()/(getTickCount() - time);
        cout <<"fps"<<fps<<endl;
    }

    return 0;
}

void ycbcrUpdate(const Mat& IM_result_cbcr,const Mat& IM_bri_T ,Mat& IM_result_cbcr_re)
{
   vector<Mat> channelsOfIM;

//    cout <<"IM_result_cbcr.type():"<<IM_result_cbcr.type()<<endl;
    split(IM_result_cbcr,channelsOfIM);
//    cout <<"channelsOfIM[0].type():"<<channelsOfIM[0].type()<<endl;
//    cout <<"IM_bri_T.type():"<<IM_bri_T.type()<<endl;
    Mat IM_bri_T_8U;
    IM_bri_T.convertTo(IM_bri_T_8U,CV_8UC1);
    channelsOfIM[0] = IM_bri_T_8U;
    merge(channelsOfIM,IM_result_cbcr_re);
    cvtColor(IM_result_cbcr_re,IM_result_cbcr_re,COLOR_YCrCb2BGR);

}

vector<Mat> extractConvMat()
{
    const string& str = "/home/pmj-nano/Desktop/1103/";
    vector<vector<double>> HVSFT;
    HVSFT.resize(15);
//    unsigned int counter = 0;
    for(int i = 1;i< 16;i++)
    {
        ifstream dataFile(str+to_string(i)+".txt");
        double dataElement;
        while(dataFile >> dataElement)
        {
//            counter++;
            HVSFT[i-1].push_back(dataElement);
        }
        dataFile.close();
//        cout <<"i:"<<i<<" "<<"counter:"<<counter<<endl;
//        counter = 0;
    }

    vector<Mat> HVSFTMAT;
    for(int i = 0; i < 15; i++)
    {
        HVSFTMAT.push_back(Mat(85,85,CV_64FC1,HVSFT[i].data()));
    }
    return HVSFTMAT;
}
