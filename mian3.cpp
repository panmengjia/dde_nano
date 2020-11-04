
#include <iostream>
#include <vector>        //提供向量头文件
#include <algorithm>     // 算法头文件，提供迭代器
#include <fstream>       //提供文件头文件
#include <iomanip>       //C++输出精度控制需要
#include <opencv.hpp>
#include <string>
using namespace cv;

using namespace std;
vector<Mat> putMat(const string& str);

int maintxt()
{

   vector<Mat> HVSFTMAT = putMat("/home/pmj-nano/Desktop/1103/");
   for(int i = 0; i < 85; i++)
   {
       cout <<i<<"          "<<endl;
       for(int j = 0; j < 85; j++)
       {
           cout <<HVSFTMAT[14].at<double>(i,j)<<" ";
       }
       cout <<endl;

   }



   return 0;
}

vector<Mat> putMat(const string& str)
{
    vector<vector<double>> HVSFT;
    HVSFT.resize(15);
    unsigned int counter = 0;
    for(int i = 1;i< 16;i++)
    {
        ifstream dataFile(str+to_string(i)+".txt");
        double dataElement;
        while(dataFile >> dataElement)
        {
            counter++;
            HVSFT[i-1].push_back(dataElement);
        }
        dataFile.close();
        cout <<"i:"<<i<<" "<<"counter:"<<counter<<endl;
        counter = 0;
    }

    vector<Mat> HVSFTMAT;
    for(int i = 0; i < 15; i++)
    {
        HVSFTMAT.push_back(Mat(85,85,CV_64FC1,HVSFT[i].data()));
    }
    return HVSFTMAT;
}
