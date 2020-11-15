
#include <iostream>
#include <vector>        //提供向量头文件
#include <algorithm>     // 算法头文件，提供迭代器
#include <fstream>       //提供文件头文件
#include <iomanip>       //C++输出精度控制需要
#include <opencv.hpp>
#include <string>

using namespace cv;
using namespace std;


void putMat(const string& str,vector<Mat>& HVSFTMAT);
vector<Mat> putMat2(const string& str);

#define FILTER_WIDTH (85)
#define FILTER_HEIGHT (85)

int main()
{

   vector<Mat> HVSFTMAT;
//   putMat("/home/pmj-nano/Desktop/1103/",HVSFTMAT);
   HVSFTMAT = putMat2("/home/pmj-nano/Desktop/1103project/");
   cout <<"HVSFTMAT.size():"<<HVSFTMAT.size()<<endl;
   for(int i = 0; i < 85; i++)
   {
       cout <<i<<"    lf      "<<endl;
       for(int j = 0; j < 85; j++)
       {
//           cout <<((double*)HVSFTMAT[5].data)[i*85+j]<<" ";
           cout <<HVSFTMAT[5].at<double>(i,j)<<" ";
       }
       cout <<endl;
   }
   return 0;
}


void putMat(const string& str,vector<Mat>& HVSFTMAT)
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
   for(int i = 0; i < 85*85; i++)
   {
        cout << HVSFT[5][i] <<" ";
   }
    cout <<"HVSFT[5].size():"<<HVSFT[5].size()<<endl;

//    vector<Mat> HVSFTMAT;
    for(int i = 0; i < 15; i++)
    {
        HVSFTMAT.push_back(Mat(85,85,CV_64FC1,HVSFT[i].data()));
    }
//    for(int i = 0; i < 85; i++)
//    {
//        cout <<i<<"          "<<endl;
//        for(int j = 0; j < 85; j++)
//        {
//            cout <<HVSFTMAT[5].at<double>(i,j)<<" ";
//        }
//        cout <<endl;

//    }
//    return HVSFTMAT;
}


//清除 clear erase swap pop_back
vector<Mat> putMat2(const string& str)
{
    vector<vector<double>> dataAllFile;
    vector<double> dataPerFile;
    double dataElement;
    for(int i = 0; i < 15; i++)
    {
        ifstream dataPerFileStream(str+to_string( i + 1) +".txt");
        while(dataPerFileStream >> dataElement)
        {
          dataPerFile.push_back(dataElement);
//          cout <<dataElement<<endl;
        }
        cout <<"dataPerFile.size() = "<<dataPerFile.size()<<endl;  //7225 right
        dataAllFile.push_back(dataPerFile);
//        dataPerFile.clear();
//        vector<double>().swap(dataPerFile); //release the memory of vector
//        dataPerFile.swap(vector<double>());
        vector<double>().swap(dataPerFile);//为什么使用，突然return vector<Mat> 就可以把数据回传过去了
        cout <<"dataPerFile.size() = "<<dataPerFile.size()<<endl;  //0 desired
        cout <<"dataPerFile.capacity() = "<<dataPerFile.capacity()<<endl;  // 8192
        //clear之后size()为0，但是capacity()不变，不为0
        cout <<"sizeof(datPerFile) = "<<sizeof(dataPerFile)<<endl;  //display 24
//        dataPerFile.swap(vector<double>());
        cout <<"dataAllFile[i].size() = "<<dataAllFile[i].size()<<endl;  //7225 为什么还是
        cout <<"dataAllFile[i].capacity() = "<<dataAllFile[i].capacity()<<endl;  //为什么还是7225
   }
    cout <<"dataAllFile.size() = "<<dataAllFile.size()<<endl;  //15
    cout <<"dataAllFile.capacity() = "<<dataAllFile.capacity()<<endl;  //16
//    for(int i = 0; i < 85*85; i++)
//    {
//        cout <<dataAllFile[5][i]<<" ";
//    }
    vector<Mat> matAllFile;
    for(int i = 0; i < 15; i++)
    {
       matAllFile.push_back(Mat(85,85,CV_64FC1,dataAllFile[i].data()));
    }
    return matAllFile;
}
