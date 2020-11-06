#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <thread>
//Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
//Helper functions for CUDA
//#include "/usr/local/cuda-9.0/samples/common/inc/helper_functions.h"
//#include "/usr/local/cuda-9.0/samples/common/inc/helper_cuda.h"
#include <iostream>
//#include "stdafx.h"
#include <algorithm>
#include <fstream>
//#include "windows.h"
#include <opencv2/opencv.hpp>
//#include  <fftw3.h>
#include "kernel.h"
double t1;

using namespace std;
using namespace cv;
std::vector <cv::Mat> rgbx_channel, rgby_channel, output;


cv::Mat binaryMaskFloat, binaryMaskFloatInverted;

std::vector<float> filter_X, filter_Y;

uchar *host_b;
uchar3 *dev_destination;
uchar *dev_dest;
uchar *dev_src;
uchar3 *dev_destinationsplit;
uchar3 *dev_patchsplit;
uchar3 *dev_patch;


float3  *destinationGradientX;
float3  *destinationGradientY;
float3  *patchGradientX;
float3  *patchGradientY;

float3  *dev_lax;
float3  *dev_lay;
float3  *laplacianX;
float3  *laplacianY;
float3  *dev_laplacianX;
float3  *dev_laplacianY;
float3  *dev_lap;
float3  *dev_boundary_points;
float3  *mod_diff;
float2 *mod_rowtemp[3];
float2 *mod_coltemp[3];
float3  *mod_ROI;
float *dev_filter_X;
float *dev_filter_Y;
cufftHandle  plan;
cufftHandle  plan2;
float *dev_binaryMaskFloat;
float *dev_binaryMaskFloatInverted;

int onetime = 0;
int onetime2 = 0;

int secondtime = 0;
int secondtime2 = 0;
int secondtime3 = 0;
int thridtime = 0;
int fourtime = 0;
dim3 blockSize(16, 16);
__global__ void Bayersplit1(uchar *v1, uchar3  *v2, int Height, int Width, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offsetR = 2 * Idx * Width + 2 * Idy;
    int offsetB = (2 * Idx + 1)* Width + (2 * Idy + 1);
    int offsetG1 = 2 * Idx * Width + (2 * Idy + 1);
    int offsetG2 = (2 * Idx + 1) * Width + 2 * Idy;
    v2[offset].x = v1[offsetR];
    v2[offset].y = v1[offsetB];
    v2[offset].z = (v1[offsetG1]+v1[offsetG2])/2;
}
__global__ void Bayersplit2(uchar *v1, uchar3  *v2,  int Height, int Width, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offsetR = 4 * Idx * Width + 4 * Idy;
    int offsetB = (4 * Idx + 1)* Width + (4 * Idy + 1);
    int offsetG1 = 4 * Idx * Width + (4 * Idy + 1);
    int offsetG2 = (4 * Idx + 1) * Width + 4 * Idy;
    v2[offset].x = (v1[offsetR] + v1[offsetR + 2]
                    + v1[offsetR + 2 * Width] + v1[offsetR + 2 + 2 * Width]) / 4;
    v2[offset].y = (v1[offsetB] + v1[offsetB + 2]
                    + v1[offsetB + 2 * Width] + v1[offsetB + 2 + 2 * Width]) / 4;
    v2[offset].z = (v1[offsetG1] + v1[offsetG1 + 2]
                    + v1[offsetG1 + 2 * Width] + v1[offsetG1 + 2 + 2 * Width]
                    + v1[offsetG2] + v1[offsetG2 + 2]
                    + v1[offsetG2 + 2 * Width] + v1[offsetG2 + 2 + 2 * Width]) / 8;
}
__global__ void Bayersplit4(uchar *v1, uchar3  *v2,  int Height, int Width, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offsetR = 8 * Idx * Width + 8 * Idy;
    int offsetB = (8 * Idx + 1)* Width + (8 * Idy + 1);
    int offsetG1 = 8 * Idx * Width + (8 * Idy + 1);
    int offsetG2 = (8 * Idx + 1) * Width + 8 * Idy;
    v2[offset].x = v1[offsetR];
    v2[offset].y = v1[offsetB];
    v2[offset].z = (v1[offsetG1] + v1[offsetG2])/2;
//    v2[offset].x = (v1[offsetR] + v1[offsetR + 2]
//            + v1[offsetR + 4] + v1[offsetR + 6]
//            + v1[offsetR + 2 * Width] + v1[offsetR + 2 + 2 * Width]
//            + v1[offsetR + 4 + 2 * Width] + v1[offsetR + 6 + 2 * Width]) / 8;
//    v2[offset].y = (v1[offsetB] + v1[offsetB + 2]
//            + v1[offsetB + 4] + v1[offsetB + 6]
//            + v1[offsetB + 2 * Width] + v1[offsetB + 2 + 2 * Width]
//            + v1[offsetB + 4 + 2 * Width] + v1[offsetB + 6 + 2 * Width]) / 8;
//    v2[offset].z = (v1[offsetG1] + v1[offsetG1 + 2]
//            + v1[offsetG1 + 4] + v1[offsetG1 + 6]
//            + v1[offsetG1 + 2 * Width] + v1[offsetG1 + 2 + 2 * Width]
//            + v1[offsetG1 + 4 + 2 * Width] + v1[offsetG1 + 6 + 2 * Width]
//            + v1[offsetG2] + v1[offsetG2 + 2]
//            + v1[offsetG2 + 4] + v1[offsetG2 + 6]
//            + v1[offsetG2 + 2 * Width] + v1[offsetG2 + 2 + 2 * Width]
//            + v1[offsetG2 + 4 + 2 * Width] + v1[offsetG2 + 6 + 2 * Width]) / 16;
}
__global__ void divergance(uchar3 *v1 , uchar3 *v2, float  *v3, float *v4,float3  *v5, float3 *v6,int Height, int Width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * Width + Idy;
    if (Idx > Height - 1 || Idy > Width - 1)return;
    if(v3[offset]==1)    //dest
        {
            if (Idy == Width - 1)
            {
                v5[offset].x = 0;
                v5[offset].y = 0;
                v5[offset].z = 0;
            }
            else
            {
                v5[offset].x = v1[offset + 1].x - v1[offset].x;
                v5[offset].y = v1[offset + 1].y - v1[offset].y;
                v5[offset].z = v1[offset + 1].z - v1[offset].z;
            }
            if (Idx == Height - 1)
            {
                v6[offset].x = 0;
                v6[offset].y = 0;
                v6[offset].z = 0;
            }
            else
            {
                v6[offset].x = v1[offset + Width].x - v1[offset].x;
                v6[offset].y = v1[offset + Width].y - v1[offset].y;
                v6[offset].z = v1[offset + Width].z - v1[offset].z;
            }
        }
     else if(v4[offset]==1)
        {
            if (Idy == Width - 1)
            {
                v5[offset].x = 0;
                v5[offset].y = 0;
                v5[offset].z = 0;
            }
            else
            {
                v5[offset].x = v2[offset + 1].x - v2[offset].x;
                v5[offset].y = v2[offset + 1].y - v2[offset].y;
                v5[offset].z = v2[offset + 1].z - v2[offset].z;
            }
            if (Idx == Height - 1)
            {
                v6[offset].x = 0;
                v6[offset].y = 0;
                v6[offset].z = 0;
            }
            else
            {
                v6[offset].x = v2[offset + Width].x - v2[offset].x;
                v6[offset].y = v2[offset + Width].y - v2[offset].y;
                v6[offset].z = v2[offset + Width].z - v2[offset].z;
            }
        }
     else
        {
         v5[offset].x = 0;
         v5[offset].y = 0;
         v5[offset].z = 0;
         v6[offset].x = 0;
         v6[offset].y = 0;
         v6[offset].z = 0;
         }
}
__global__ void laplacian_add(float3  *v1, float3  *v2, float3  *v3,int Height, int Width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * Width + Idy;
    if (Idx > Height - 1 || Idy > Width - 1)return;
    if (Idx != 0 && Idy != 0)
    {
        v3[offset].x = v2[offset].x - v2[offset - Width].x
                     + v1[offset].x - v1[offset - 1].x;
        v3[offset].y = v2[offset].y - v2[offset - Width].y
                     + v1[offset].y - v1[offset - 1].y;
        v3[offset].z = v2[offset].z - v2[offset - Width].z
                     + v1[offset].z - v1[offset - 1].z;
     }
}
__global__ void gpu_rectangle(uchar3  *v1, uchar3  *v2, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    if (Idx < height - 1 && Idx > 0 && Idy < width - 1 && Idy > 0)
    {
//        v2[offset] = 0;
        v2[offset].x = 0;
        v2[offset].y = 0;
        v2[offset].z = 0;
    }
    else
    {
//        v2[offset] = v1[offset];
        v2[offset].x = v1[offset].x;
        v2[offset].y = v1[offset].y;
        v2[offset].z = v1[offset].z;
    }
}
__global__ void gpu_Laplacian(uchar3  *v1, float3  *v2, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    if (Idx < height - 1 && Idx > 0 && Idy < width - 1 && Idy > 0)
    {
//        v2[offset] = v1[offset + 1] + v1[offset - 1] + v1[offset - width] + v1[offset + width] - 4 * v1[offset];
        v2[offset].x = v1[offset + 1].x + v1[offset - 1].x + v1[offset - width].x + v1[offset + width].x - 4 * v1[offset].x;
        v2[offset].y = v1[offset + 1].y + v1[offset - 1].y + v1[offset - width].y + v1[offset + width].y - 4 * v1[offset].y;
        v2[offset].z = v1[offset + 1].z + v1[offset - 1].z + v1[offset - width].z + v1[offset + width].z - 4 * v1[offset].z;
    }
}
__global__ void gpu_rectangle_Laplacian(uchar3  *v1, float3  *v2, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    if (Idx < height - 1 && Idx > 0 && Idy < width - 1 && Idy > 0)
    if (Idx == 1)   //up
    {
        v2[offset].x = v1[offset+width].x;
        v2[offset].y = v1[offset+width].y;
        v2[offset].z = v1[offset+width].z;
    }
    else if (Idx == height - 2)   //down
    {
//        v2[offset] = v1[offset];
        v2[offset].x = v1[offset-width].x;
        v2[offset].y = v1[offset-width].y;
        v2[offset].z = v1[offset-width].z;
    }
    else if (Idy == 1)    //left
    {
        v2[offset].x = v1[offset-1].x;
        v2[offset].y = v1[offset-1].y;
        v2[offset].z = v1[offset-1].z;
    }
    else if(Idy ==width -2 )   //right
    {
        v2[offset].x = v1[offset+1].x;
        v2[offset].y = v1[offset+1].y;
        v2[offset].z = v1[offset+1].z;
    }
}
__global__ void gpu_mod_diff(float3  *v1, float3  *v2, float3  *v3, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offset = Idx * width + Idy;
    int offset2 = (Idx - 1)*(width - 2) + Idy - 1;
    if (Idx > height - 2 || Idy > width - 2 || Idx<1 || Idy<1)return;
    {
//        v3[offset2] = v1[offset] - v2[offset];
        v3[offset2].x = v1[offset].x - v2[offset].x;
        v3[offset2].y = v1[offset].y - v2[offset].y;
        v3[offset2].z = v1[offset].z - v2[offset].z;
    }
}
__global__ void gpu_rectangle_Laplacian_diff(uchar3 *v1, float3 *v2, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    if (Idx < height - 1 && Idx > 0 && Idy < width - 1 && Idy > 0)
    {
        if (Idx == height - 2 || Idx == 1  || Idy == width - 2 || Idy == 1 )
        {
        v2[offset].x = v2[offset].x - v1[offset + 1].x - v1[offset - 1].x
                     - v1[offset - width].x - v1[offset + width].x;
        v2[offset].y = v2[offset].y - v1[offset + 1].y - v1[offset - 1].y
                     - v1[offset - width].y - v1[offset + width].y;
        v2[offset].z = v2[offset].z - v1[offset + 1].z - v1[offset - 1].z
                     - v1[offset - width].z - v1[offset + width].z;
        }
    }
}
__global__ void gpu_mod_rowtemp(float3  *v1, float2 *v2,float2 *v3, float2 *v4, int height, int width, int width_kenerl)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offset = Idx * width + Idy;
    int offset1 = Idx*width_kenerl + Idy - 1;
    int offset2 = Idx*width_kenerl + 2 * width_kenerl + 1 - Idy;
    if (Idy == 0 || Idy == (width_kenerl + 1))
    {
        v2[offset].x = 0;
        v2[offset].y = 0;
        v3[offset].x = 0;
        v3[offset].y = 0;
        v4[offset].x = 0;
        v4[offset].y = 0;
    }
    if (Idy <= width_kenerl && Idy > 0)
    {
        v2[offset].x = v1[offset1].x;
        v2[offset].y = 0;
        v3[offset].x = v1[offset1].y;
        v3[offset].y = 0;
        v4[offset].x = v1[offset1].z;
        v4[offset].y = 0;
    }
    if (Idy > (width_kenerl + 1))
    {
        v2[offset].x = -v1[offset2].x;
        v2[offset].y = 0;
        v3[offset].x = -v1[offset2].y;
        v3[offset].y = 0;
        v4[offset].x = -v1[offset2].z;
        v4[offset].y = 0;
    }
}
__global__ void gpu_mod_coltemp(float2 *v2, float2 *v3, float2 *v4, float2 *v5,float2 *v6,float2 *v7, int height, int width, int width_kenerl, int height_kenerl)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offset = Idx * width + Idy;
    int offset1 = (Idy - 1)*width_kenerl + Idx + 1;
    int offset2 = (2 * height_kenerl + 1 - Idy)*width_kenerl + Idx + 1;
    if (Idy == 0 || Idy == (height_kenerl + 1))
    {
        v5[offset].x = 0;
        v5[offset].y = 0;
        v6[offset].x = 0;
        v6[offset].y = 0;
        v7[offset].x = 0;
        v7[offset].y = 0;
    }
    if (Idy <= height_kenerl && Idy > 0)
    {
        v5[offset].x = v2[offset1].y;
        v5[offset].y = 0;
        v6[offset].x = v3[offset1].y;
        v6[offset].y = 0;
        v7[offset].x = v4[offset1].y;
        v7[offset].y = 0;
    }
    if (Idy > (height_kenerl + 1))
    {
        v5[offset].x = -v2[offset2].y;
        v5[offset].y = 0;
        v6[offset].x = -v3[offset2].y;
        v6[offset].y = 0;
        v7[offset].x = -v4[offset2].y;
        v7[offset].y = 0;
    }
}
__global__ void gpu_mod_coltemp2(float2 *v2, float2 *v3, float2 *v4, float2 *v5, float2 *v6, float2 *v7, int height, int width, int width_kenerl, int height_kenerl)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offset = Idx * width + Idy;
    int offset1 = (Idy - 1)*width_kenerl + Idx + 1;
    int offset2 = (2 * height_kenerl + 1 - Idy)*width_kenerl + Idx + 1;
    if (Idy == 0 || Idy == (height_kenerl + 1))
    {
        v5[offset].x = 0;
        v5[offset].y = 0;
        v6[offset].x = 0;
        v6[offset].y = 0;
        v7[offset].x = 0;
        v7[offset].y = 0;
    }
    if (Idy <= height_kenerl && Idy > 0)
    {
        v5[offset].x = v2[offset1].y / width_kenerl;
        v5[offset].y = 0;
        v6[offset].x = v3[offset1].y/ width_kenerl;
        v6[offset].y = 0;
        v7[offset].x = v4[offset1].y/ width_kenerl;
        v7[offset].y = 0;
    }

    if (Idy > (height_kenerl + 1))
    {
        v5[offset].x = -v2[offset2].y / width_kenerl;
        v5[offset].y = 0;
        v6[offset].x = -v3[offset2].y/ width_kenerl;
        v6[offset].y = 0;
        v7[offset].x = -v4[offset2].y/ width_kenerl;
        v7[offset].y = 0;
    } 
}
__global__ void gpu_mod_ROI(float2 *v2, float2 *v3, float2 *v4,float3  *v5, float *v6, float *v7, int height, int width, int width_kenerl)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offset = Idx * width + Idy;
    int offset1 = Idy *width_kenerl + Idx + 1;
    v5[offset].x = v2[offset1].y/(v6[Idy] + v7[Idx] - 4);
    v5[offset].y = v3[offset1].y/(v6[Idy] + v7[Idx] - 4);
    v5[offset].z = v4[offset1].y/(v6[Idy] + v7[Idx] - 4);
}
__global__ void gpu_mod_ROI2(float2 *v2, float2 *v3, float2 *v4, uchar3 *v5, int height, int width, int width_kenerl)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offset = (Idx + 1) * (width + 2) + Idy + 1;
    int offset1 = Idy *width_kenerl + Idx + 1;
    v5[offset].x = v2[offset1].y / width_kenerl;
    if (v5[offset].x < 0)
        v5[offset].x = 0;
    if (v5[offset].x > 255)
        v5[offset].x = 255;
    v5[offset].y = v3[offset1].y / width_kenerl;
    if (v5[offset].y < 0)
        v5[offset].y = 0;
    if (v5[offset].y > 255)
        v5[offset].y = 255;
    v5[offset].z = v4[offset1].y / width_kenerl;
    if (v5[offset].z < 0)
        v5[offset].z = 0;
    if (v5[offset].z > 255)
        v5[offset].z = 255;
}
__global__ void gpu_mod_ROI3(float2 *v2,float2 *v3, float2 *v4, float3 *v5, int height, int width, int width_kenerl)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offset = Idx  * width + Idy;
    int offset1 = Idy *width_kenerl + Idx + 1;
    v5[offset].x = v2[offset1].y / width_kenerl;
    v5[offset].y = v3[offset1].y / width_kenerl;
    v5[offset].z = v4[offset1].y / width_kenerl;
}
__global__ void gpu_mod_ROI4(float3 *v2, uchar3 *v5, int height, int width, int width_kenerl)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offset = Idx  * width + Idy;
    int offset1 = (Idx + 1) *(width + 2) + Idy + 1;
    if (v2[offset].x < 0)
        v5[offset1].x = 0;
    else if (v2[offset].x > 255)
        v5[offset1].x = 255;
    else
        v5[offset1].x = v2[offset].x;
    if (v2[offset].y < 0)
    v5[offset1].y = 0;
    else if (v2[offset].y > 255)
    v5[offset1].y = 255;
    else
    v5[offset1].y = v2[offset].y;
    if (v2[offset].z < 0)
    v5[offset1].z = 0;
    else if (v2[offset].z > 255)
    v5[offset1].z = 255;
    else
    v5[offset1].z = v2[offset].z;
}
__global__ void dianchu(float2 *v1, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    v1[offset].x = v1[offset].x / width;
    v1[offset].y = -v1[offset].y / width;
}
__global__ void Bayer_merge1(uchar *v1, uchar3 *v2, int Height, int Width, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offsetR = 2 * Idx * Width + 2 * Idy;
    int offsetB = (2 * Idx + 1)* Width + (2 * Idy + 1);
    int offsetG1 = 2 * Idx * Width + (2 * Idy + 1);
    int offsetG2 = (2 * Idx + 1) * Width + 2 * Idy;
        v1[offsetR] = v2[offset].x;
        v1[offsetB] = v2[offset].y;
        v1[offsetG1] = v2[offset].z;
        v1[offsetG2] = v2[offset].z;

}
__global__ void Bayer_merge2(uchar *v1, uchar3  *v2, int Height, int Width, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offsetR = 4 * Idx * Width + 4 * Idy;
    int offsetB = (4 * Idx + 1)* Width + (4 * Idy + 1);
    int offsetG1 = 4 * Idx * Width + (4 * Idy + 1);
    int offsetG2 = (4 * Idx + 1) * Width + 4 * Idy;
    if(Idx == height - 1 || Idy == width - 1)
    {
        v1[offsetR] = v2[offset].x;
        v1[offsetR + 2] = v2[offset].x;
        v1[offsetR + 2 * Width] = v2[offset].x;
        v1[offsetR + 2 + 2 * Width] = v2[offset].x;
        v1[offsetB] = v2[offset].y;
        v1[offsetB + 2] = v2[offset].y;
        v1[offsetB + 2 * Width] = v2[offset].y;
        v1[offsetB + 2 + 2 * Width] = v2[offset].y;
        v1[offsetG1] = v2[offset].z;
        v1[offsetG1 + 2] = v2[offset].z;
        v1[offsetG1 + 2 * Width] = v2[offset].z;
        v1[offsetG1 + 2 + Width] = v2[offset].z;
        v1[offsetG2] = v2[offset].z;
        v1[offsetG2 + 2] = v2[offset].z;
        v1[offsetG2 + 2 * Width] = v2[offset].z;
        v1[offsetG2 + 2 + 2 * Width] = v2[offset].z;
    }
    else
    {
        v1[offsetR] = v2[offset].x;
        v1[offsetR + 2] = (v2[offset].x + v2[offset + 1].x)/2;
        v1[offsetR + 2 * Width] = (v2[offset].x + v2[offset + width].x)/2;
        v1[offsetR + 2 + 2 * Width] = (v2[offset].x + v2[offset+1].x
                                     + v2[offset+width].x + v2[offset + 1 + width].x)/4;

        v1[offsetB] = v2[offset].y;
        v1[offsetB + 2] = (v2[offset].y + v2[offset + 1].y)/2;
        v1[offsetB + 2 * Width] = (v2[offset].y + v2[offset + width].y)/2;
        v1[offsetB + 2 + 2 * Width] = (v2[offset].y + v2[offset+1].y
                                     + v2[offset+width].y + v2[offset + 1 + width].y)/4;

        v1[offsetG1] = v2[offset].z;
        v1[offsetG1 + 2] = (v2[offset].z + v2[offset + 1].z)/2;
        v1[offsetG1 + 2 * Width] = (v2[offset].z + v2[offset + width].z)/2;
        v1[offsetG1 + 2 + 2 * Width] = (v2[offset].z + v2[offset+1].z
                                      + v2[offset+width].z + v2[offset + 1 + width].z)/4;

        v1[offsetG2] = v2[offset].z;
        v1[offsetG2 + 2] = (v2[offset].z + v2[offset + 1].z)/2;
        v1[offsetG2 + 2 * Width] = (v2[offset].z + v2[offset + width].z)/2;
        v1[offsetG2 + 2 + 2 * Width] = (v2[offset].z + v2[offset+1].z
                                      + v2[offset+width].z + v2[offset + 1 + width].z)/4;
    }
}
__global__ void Bayer_merge4(uchar *v1, uchar3  *v2, int Height, int Width, int height, int width)
{
    int Idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = Idx * width + Idy;
    if (Idx > height - 1 || Idy > width - 1)return;
    int offsetR = 8 * Idx * Width + 8 * Idy;
    int offsetB = (8 * Idx + 1) * Width + (8 * Idy + 1);
    int offsetG1 = 8 * Idx * Width + (8 * Idy + 1);
    int offsetG2 = (8 * Idx + 1) * Width + 8 * Idy;
    if (Idx == height - 1 || Idy == width - 1)
    {
        v1[offsetR    ] = v2[offset].x;
        v1[offsetR + 2] = (v2[offset].x);
        v1[offsetR + 4] = v2[offset].x;
        v1[offsetR + 6] = v2[offset].x;
        v1[offsetR +     2 * Width] = v2[offset].x;
        v1[offsetR + 2 + 2 * Width] = v2[offset].x;
        v1[offsetR + 4 + 2 * Width] = v2[offset].x;
        v1[offsetR + 6 + 2 * Width] = v2[offset].x;
        v1[offsetR +     4 * Width] = v2[offset].x;
        v1[offsetR + 2 + 4 * Width] = v2[offset].x;
        v1[offsetR + 4 + 4 * Width] = v2[offset].x;
        v1[offsetR + 6 + 4 * Width] = v2[offset].x;
        v1[offsetR +     6 * Width] = v2[offset].x;
        v1[offsetR + 2 + 6 * Width] = v2[offset].x;
        v1[offsetR + 4 + 6 * Width] = v2[offset].x;
        v1[offsetR + 6 + 6 * Width] = v2[offset].x;

        v1[offsetB    ] = v2[offset].y;
        v1[offsetB + 2] = v2[offset].y;
        v1[offsetB + 4] = v2[offset].y;
        v1[offsetB + 6] = v2[offset].y;
        v1[offsetB +     2 * Width] = v2[offset].y;
        v1[offsetB + 2 + 2 * Width] = v2[offset].y;
        v1[offsetB + 4 + 2 * Width] = v2[offset].y;
        v1[offsetB + 6 + 2 * Width] = v2[offset].y;
        v1[offsetB +     4 * Width] = v2[offset].y;
        v1[offsetB + 2 + 4 * Width] = v2[offset].y;
        v1[offsetB + 4 + 4 * Width] = v2[offset].y;
        v1[offsetB + 6 + 4 * Width] = v2[offset].y;
        v1[offsetB +     6 * Width] = v2[offset].y;
        v1[offsetB + 2 + 6 * Width] = v2[offset].y;
        v1[offsetB + 4 + 6 * Width] = v2[offset].y;
        v1[offsetB + 6 + 6 * Width] = v2[offset].y;

        v1[offsetG1    ] = v2[offset].z;
        v1[offsetG1 + 2] = v2[offset].z;
        v1[offsetG1 + 4] = v2[offset].z;
        v1[offsetG1 + 6] = v2[offset].z;
        v1[offsetG1 +     2 * Width] = v2[offset].z;
        v1[offsetG1 + 2 + 2 * Width] = v2[offset].z;
        v1[offsetG1 + 4 + 2 * Width] = v2[offset].z;
        v1[offsetG1 + 6 + 2 * Width] = v2[offset].z;
        v1[offsetG1 +     4 * Width] = v2[offset].z;
        v1[offsetG1 + 2 + 4 * Width] = v2[offset].z;
        v1[offsetG1 + 4 + 4 * Width] = v2[offset].z;
        v1[offsetG1 + 6 + 4 * Width] = v2[offset].z;
        v1[offsetG1 +     6 * Width] = v2[offset].z;
        v1[offsetG1 + 2 + 6 * Width] = v2[offset].z;
        v1[offsetG1 + 4 + 6 * Width] = v2[offset].z;
        v1[offsetG1 + 6 + 6 * Width] = v2[offset].z;

        v1[offsetG2    ] = v2[offset].z;
        v1[offsetG2 + 2] = v2[offset].z;
        v1[offsetG2 + 4] = v2[offset].z;
        v1[offsetG2 + 6] = v2[offset].z;
        v1[offsetG2 +     2 * Width] = v2[offset].z;
        v1[offsetG2 + 2 + 2 * Width] = v2[offset].z;
        v1[offsetG2 + 4 + 2 * Width] = v2[offset].z;
        v1[offsetG2 + 6 + 2 * Width] = v2[offset].z;
        v1[offsetG2 +     4 * Width] = v2[offset].z;
        v1[offsetG2 + 2 + 4 * Width] = v2[offset].z;
        v1[offsetG2 + 4 + 4 * Width] = v2[offset].z;
        v1[offsetG2 + 6 + 4 * Width] = v2[offset].z;
        v1[offsetG2 +     6 * Width] = v2[offset].z;
        v1[offsetG2 + 2 + 6 * Width] = v2[offset].z;
        v1[offsetG2 + 4 + 6 * Width] = v2[offset].z;
        v1[offsetG2 + 6 + 6 * Width] = v2[offset].z;
    }
    else
    {
        v1[offsetR    ] = v2[offset].x;
        v1[offsetR + 2] = (3*v2[offset].x+v2[offset+1].x)/4;
        v1[offsetR + 4] = (v2[offset].x+v2[offset+1].x)/2;
        v1[offsetR + 6] = (v2[offset].x+3*v2[offset+1].x)/4;
        v1[offsetR +     2 * Width] = (3*v2[offset].x+v2[offset+width].x)/4;
        v1[offsetR + 2 + 2 * Width] = (9*v2[offset].x+3*v2[offset+1].x+3*v2[offset+width].x+v2[offset+width+1].x)/16;
        v1[offsetR + 4 + 2 * Width] = (3*v2[offset].x+3*v2[offset+1].x+v2[offset+width].x+v2[offset+width+1].x)/8;
        v1[offsetR + 6 + 2 * Width] = (3*v2[offset].x+9*v2[offset+1].x+v2[offset+width].x+3*v2[offset+width+1].x)/16;
        v1[offsetR +     4 * Width] = (v2[offset].x+v2[offset+width].x)/2;
        v1[offsetR + 2 + 4 * Width] = (3*v2[offset].x+v2[offset+1].x+3*v2[offset+width].x+v2[offset+width+1].x)/8;
        v1[offsetR + 4 + 4 * Width] = (v2[offset].x+v2[offset+1].x+v2[offset+width].x+v2[offset+width+1].x)/4;
        v1[offsetR + 6 + 4 * Width] = (v2[offset].x+3*v2[offset+1].x+v2[offset+width].x+3*v2[offset+width+1].x)/8;
        v1[offsetR +     6 * Width] = (v2[offset].x+3*v2[offset+width].x)/4;
        v1[offsetR + 2 + 6 * Width] = (3*v2[offset].x+v2[offset+1].x+9*v2[offset+width].x+v2[offset+width+1].x)/16;
        v1[offsetR + 4 + 6 * Width] = (v2[offset].x+v2[offset+1].x+3*v2[offset+width].x+3*v2[offset+width+1].x)/8;
        v1[offsetR + 6 + 6 * Width] = (v2[offset].x+3*v2[offset+1].x+3*v2[offset+width].x+9*v2[offset+width+1].x)/16;

        v1[offsetB    ] = v2[offset].y;
        v1[offsetB + 2] = (3*v2[offset].y+v2[offset+1].y)/4;
        v1[offsetB + 4] = (v2[offset].y+v2[offset+1].y)/2;
        v1[offsetB + 6] = (v2[offset].y+3*v2[offset+1].y)/4;
        v1[offsetB +     2 * Width] = (3*v2[offset].y+v2[offset+width].y)/4;
        v1[offsetB + 2 + 2 * Width] = (9*v2[offset].y+3*v2[offset+1].y+3*v2[offset+width].y+v2[offset+width+1].y)/16;
        v1[offsetB + 4 + 2 * Width] = (3*v2[offset].y+3*v2[offset+1].y+v2[offset+width].y+v2[offset+width+1].y)/8;
        v1[offsetB + 6 + 2 * Width] = (3*v2[offset].y+9*v2[offset+1].y+v2[offset+width].y+3*v2[offset+width+1].y)/16;
        v1[offsetB +     4 * Width] = (v2[offset].y+v2[offset+width].y)/2;
        v1[offsetB + 2 + 4 * Width] = (3*v2[offset].y+v2[offset+1].y+3*v2[offset+width].y+v2[offset+width+1].y)/8;
        v1[offsetB + 4 + 4 * Width] = (v2[offset].y+v2[offset+1].y+v2[offset+width].y+v2[offset+width+1].y)/4;
        v1[offsetB + 6 + 4 * Width] = (v2[offset].y+3*v2[offset+1].y+v2[offset+width].y+3*v2[offset+width+1].y)/8;
        v1[offsetB +     6 * Width] = (v2[offset].y+3*v2[offset+width].y)/4;
        v1[offsetB + 2 + 6 * Width] = (3*v2[offset].y+v2[offset+1].y+9*v2[offset+width].y+v2[offset+width+1].y)/16;
        v1[offsetB + 4 + 6 * Width] = (v2[offset].y+v2[offset+1].y+3*v2[offset+width].y+3*v2[offset+width+1].y)/8;
        v1[offsetB + 6 + 6 * Width] = (v2[offset].y+3*v2[offset+1].y+3*v2[offset+width].y+9*v2[offset+width+1].y)/16;

        v1[offsetG1    ] = v2[offset].z;
        v1[offsetG1 + 2] = (3*v2[offset].z+v2[offset+1].z)/4;
        v1[offsetG1 + 4] = (v2[offset].z+v2[offset+1].z)/2;
        v1[offsetG1 + 6] = (v2[offset].z+3*v2[offset+1].z)/4;
        v1[offsetG1 +     2 * Width] = (3*v2[offset].z+v2[offset+width].z)/4;
        v1[offsetG1 + 2 + 2 * Width] = (9*v2[offset].z+3*v2[offset+1].z+3*v2[offset+width].z+v2[offset+width+1].z)/16;
        v1[offsetG1 + 4 + 2 * Width] = (3*v2[offset].z+3*v2[offset+1].z+v2[offset+width].z+v2[offset+width+1].z)/8;
        v1[offsetG1 + 6 + 2 * Width] = (3*v2[offset].z+9*v2[offset+1].z+v2[offset+width].z+3*v2[offset+width+1].z)/16;
        v1[offsetG1 +     4 * Width] = (v2[offset].z+v2[offset+width].z)/2;
        v1[offsetG1 + 2 + 4 * Width] = (3*v2[offset].z+v2[offset+1].z+3*v2[offset+width].z+v2[offset+width+1].z)/8;
        v1[offsetG1 + 4 + 4 * Width] = (v2[offset].z+v2[offset+1].z+v2[offset+width].z+v2[offset+width+1].z)/4;
        v1[offsetG1 + 6 + 4 * Width] = (v2[offset].z+3*v2[offset+1].z+v2[offset+width].z+3*v2[offset+width+1].z)/8;
        v1[offsetG1 +     6 * Width] = (v2[offset].z+3*v2[offset+width].z)/4;
        v1[offsetG1 + 2 + 6 * Width] = (3*v2[offset].z+v2[offset+1].z+9*v2[offset+width].z+v2[offset+width+1].z)/16;
        v1[offsetG1 + 4 + 6 * Width] = (v2[offset].z+v2[offset+1].z+3*v2[offset+width].z+3*v2[offset+width+1].z)/8;
        v1[offsetG1 + 6 + 6 * Width] = (v2[offset].z+3*v2[offset+1].z+3*v2[offset+width].z+9*v2[offset+width+1].z)/16;

        v1[offsetG2    ] = v2[offset].z;
        v1[offsetG2 + 2] = (3*v2[offset].z+v2[offset+1].z)/4;
        v1[offsetG2 + 4] = (v2[offset].z+v2[offset+1].z)/2;
        v1[offsetG2 + 6] = (v2[offset].z+3*v2[offset+1].z)/4;
        v1[offsetG2 +     2 * Width] = (3*v2[offset].z+v2[offset+width].z)/4;
        v1[offsetG2 + 2 + 2 * Width] = (9*v2[offset].z+3*v2[offset+1].z+3*v2[offset+width].z+v2[offset+width+1].z)/16;
        v1[offsetG2 + 4 + 2 * Width] = (3*v2[offset].z+3*v2[offset+1].z+v2[offset+width].z+v2[offset+width+1].z)/8;
        v1[offsetG2 + 6 + 2 * Width] = (3*v2[offset].z+9*v2[offset+1].z+v2[offset+width].z+3*v2[offset+width+1].z)/16;
        v1[offsetG2 +     4 * Width] = (v2[offset].z+v2[offset+width].z)/2;
        v1[offsetG2 + 2 + 4 * Width] = (3*v2[offset].z+v2[offset+1].z+3*v2[offset+width].z+v2[offset+width+1].z)/8;
        v1[offsetG2 + 4 + 4 * Width] = (v2[offset].z+v2[offset+1].z+v2[offset+width].z+v2[offset+width+1].z)/4;
        v1[offsetG2 + 6 + 4 * Width] = (v2[offset].z+3*v2[offset+1].z+v2[offset+width].z+3*v2[offset+width+1].z)/8;
        v1[offsetG2 +     6 * Width] = (v2[offset].z+3*v2[offset+width].z)/4;
        v1[offsetG2 + 2 + 6 * Width] = (3*v2[offset].z+v2[offset+1].z+9*v2[offset+width].z+v2[offset+width+1].z)/16;
        v1[offsetG2 + 4 + 6 * Width] = (v2[offset].z+v2[offset+1].z+3*v2[offset+width].z+3*v2[offset+width+1].z)/8;
        v1[offsetG2 + 6 + 6 * Width] = (v2[offset].z+3*v2[offset+1].z+3*v2[offset+width].z+9*v2[offset+width+1].z)/16;
}

}


//camerafuse out;
Mat dst_mask2;
Mat result;
Mat src_channel[3];
Mat dest_channel[3];

//Mat src;
//Mat dest;
//----------------------------------------------------------------------------
Mat fuse(Mat src_ori ,Mat dest_ori,ROI_num ROI,int ones_ti,int bayer)
{

//        split(src, src1);
//    double time_start,time_end,time_diff;
//    cudaEvent_t     start, stop;
//        cudaEventCreate(&start);
//        float elapsedTime;
//        cudaEventCreate(&stop);
//        cudaEventRecord(start, 0);
//-----------------------------M1: confirm dst-----------------------------------------------

//int dst_X=0;
//int dst_Y=0;
//int dst_width=1300;
//int dst_height=1200;
//int diff=100;

//int src_X=dst_X+diff;
//int src_Y=dst_Y+diff;
//int src_width=dst_width-2*diff;
//int src_height=dst_height-2*diff;

//-----------------------------M2: confirm src-----------------------------------------------

//int src_X=600;
//int src_Y=600;
//int src_width=300;
//int src_height=300;
//int diff=200;


//int dst_X=src_X-diff;
//int dst_Y=src_X-diff;
//int dst_width=src_width+2*diff;
//int dst_height=src_height+2*diff;

//----------------------------------------------------------------------------
int dst_X=ROI.dx;
int dst_Y=ROI.dy;
int dst_width=ROI.dw;
int dst_height=ROI.dh;
int src_X=ROI.sx;
int src_Y=ROI.sy;
int src_width=ROI.sw;
int src_height=ROI.sh;
int diff=ROI.dif;
//    int dst_X=0;
//    int dst_Y=0;
//    int dst_width=1920;
//    int dst_height=1200;
//    int src_X=ROI.sx;
//    int src_Y=ROI.sy;
//    int src_width=ROI.sw;
//    int src_height=ROI.sh;
//    int diff=ROI.dif;

int minxd = src_X - dst_X;
int maxxd = minxd + src_width;
int minyd = src_Y - dst_Y;
int maxyd = minyd + src_height;
int rows = dst_height / (bayer * 2);
int cols = dst_width  / (bayer * 2);


    //int ones_ti = 0;

    Rect roi_d(dst_X, dst_Y, dst_width, dst_height);
    Rect roi_s(src_X, src_Y, src_width, src_height);
    Rect roi_mask(minxd,minyd,src_width,src_height);
    Mat dst_mask = Mat::zeros(roi_d.size(), CV_8UC1);
    Mat src_mask  = Mat::zeros(roi_d.size(), CV_8UC1);

//    Mat src_ori=imread("/home/nvidia/Desktop/image/src.bmp");
//    Mat dest_ori = imread("/home/nvidia/Desktop/image/dest.bmp");
//    time_start=(double)clock();

//------------------------split ori------------------------------------
//    split(src_ori,src_channel);
//    split(dest_ori,dest_channel);
//    src_channel[0](roi_d).copyTo(src);
//    dest_channel[0](roi_d).copyTo(dest);

//------------------------split dst-----------------------------------
    src_ori.copyTo(src_mask(roi_mask));
    Mat src=src_mask;
//    imshow("1",src);
//    waitKey(0);
    Mat dest=dest_ori(roi_d);

    split(src,src_channel);
    split(dest,dest_channel);

//------------------------------------------------------------



//    time_end=(double)clock();
//    time_diff=(time_end-time_start)/1000.0;
//    printf("imread:  %3.0f ms\n", time_diff);

//    src.copyTo(cd_mask(roi_d));
    if (ones_ti == 0)
    {

        cout<<roi_mask<<endl;
        Mat mask = Mat::zeros(roi_d.size(), CV_8UC1);
        mask(roi_mask).setTo(255);
        CV_Assert(minxd >= 0 && minyd >= 0 && maxyd <= dest.rows && maxxd <= dest.cols);//if()break;
        filter_X.resize(cols - 2);
        for (int i = 0; i < cols - 2; ++i)
            filter_X[i] = 2.0f * std::cos(static_cast<float>(CV_PI) * (i + 1) / (cols - 1));
        filter_Y.resize(rows - 2);
        for (int j = 0; j <rows - 2; ++j)
            filter_Y[j] = 2.0f * std::cos(static_cast<float>(CV_PI) * (j + 1) / (rows - 1));
        mask(roi_mask).copyTo(dst_mask(roi_mask));
        cudaMalloc((void**)&dev_dest, sizeof(uchar) * dest.rows * dest.cols);
        cudaMalloc((void**)&dev_src, sizeof(uchar) * dest.rows * dest.cols);
        cudaMalloc((void**)&dev_destinationsplit, sizeof(uchar3) * rows * cols );
        cudaMalloc((void**)&dev_patchsplit, sizeof(uchar3) * rows * cols);
        cudaMalloc((void**)&dev_destination, sizeof(uchar3) * rows * cols);
        cudaMalloc((void**)&destinationGradientX, sizeof(float3)* rows * cols);
        cudaMalloc((void**)&destinationGradientY, sizeof(float3) * rows * cols);
        //cudaMalloc((void**)&dev_patch, sizeof(uchar) * dest.rows * dest.cols);
        cudaMalloc((void**)&patchGradientX, sizeof(float3) * rows * cols);
        cudaMalloc((void**)&patchGradientY, sizeof(float3) * rows * cols);

        Mat Kernel = Mat::ones(3, 3, CV_8UC1);
        Mat dst_mast_resize = Mat::zeros(rows , cols , CV_8UC1);
        Size dsize = Size(cols , rows);
        resize(dst_mask, dst_mask, dsize);
        erode(dst_mask, dst_mask2, Kernel, Point(-1, -1),3);
        dst_mask2.convertTo(binaryMaskFloat, CV_32FC1, 1.0/255.0);
        cudaMalloc((void**)&dev_binaryMaskFloat, sizeof(float) * rows * cols);
        cudaMemcpy(dev_binaryMaskFloat, binaryMaskFloat.data, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
        bitwise_not(dst_mask, dst_mast_resize);
        dst_mast_resize.convertTo(binaryMaskFloatInverted, CV_32FC1, 1.0 / 255.0);
        cudaMalloc((void**)&dev_binaryMaskFloatInverted, sizeof(float) * rows * cols);
        cudaMemcpy(dev_binaryMaskFloatInverted, binaryMaskFloatInverted.data, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&laplacianX, sizeof(float3) * rows * cols);
        cudaMalloc((void**)&laplacianY, sizeof(float3)* rows * cols);
        cudaMalloc((void**)&dev_laplacianX, sizeof(float3) * rows * cols);
        cudaMalloc((void**)&dev_laplacianY, sizeof(float3) * rows * cols);
        cudaMalloc((void**)&dev_lap, sizeof(float3) * rows * cols);
        cudaMalloc((void**)&dev_boundary_points, sizeof(float3) * rows * cols);
//        Mat boundary_points= Mat::zeros(rows , cols , CV_32FC3);
//        cudaMemcpy(dev_boundary_points, boundary_points.data, sizeof(float3)* rows * cols, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&mod_diff, sizeof(float3) * (rows - 2) * (cols - 2));
        for (int i = 0; i < 3; i++)
        {
            cudaMalloc((void**)&mod_rowtemp[i], sizeof(float2) * (dest.rows - 2)*(2 * (dest.cols - 2) + 2));
            cudaMalloc((void**)&mod_coltemp[i], sizeof(float2) * (dest.cols - 2)*(2 * (dest.rows - 2) + 2));
        }
        cudaMalloc((void**)&mod_ROI, sizeof(float3) * (cols - 2)*(rows - 2));
        cudaMalloc((void**)&dev_filter_X, sizeof(float) * (cols - 2));
        cudaMalloc((void**)&dev_filter_Y, sizeof(float) * (rows - 2));
        cudaMemcpy(dev_filter_X, filter_X.data(), sizeof(float) * (cols - 2), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_filter_Y, filter_Y.data(), sizeof(float) * (rows - 2), cudaMemcpyHostToDevice);

        cufftPlan1d(&plan, (2 * (cols - 2) + 2), CUFFT_C2C, (rows - 2));
        cufftPlan1d(&plan2, (2 * (rows - 2) + 2), CUFFT_C2C, (cols - 2));
    }

//    time_start=(double)clock();
//------------------------split ori------------------------------------
//    cudaMemcpy(dev_src , src.data, sizeof(uchar) * dest.rows * dest.cols, cudaMemcpyHostToDevice);
//    cudaMemcpy(dev_dest, dest.data   , sizeof(uchar) * dest.rows * dest.cols, cudaMemcpyHostToDevice);

//------------------------split dst------------------------------------
    cudaMemcpy(dev_src , src_channel[0].data , sizeof(uchar) * dest.rows * dest.cols, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dest, dest_channel[0].data, sizeof(uchar) * dest.rows * dest.cols, cudaMemcpyHostToDevice);

//---------------------------------------------------------------------
    dim3 gridSize((int)ceil((float)dest.rows / (float)blockSize.x), (int)ceil((float)dest.cols / (float)blockSize.y));
    switch(bayer)
    {
    case 1:{Bayersplit1 << <gridSize, blockSize >> > (dev_src , dev_patchsplit      , dest.rows, dest.cols, rows , cols );
            Bayersplit1 << <gridSize, blockSize >> > (dev_dest, dev_destinationsplit, dest.rows, dest.cols, rows , cols );break;}
    case 2:{Bayersplit2 << <gridSize, blockSize >> > (dev_src , dev_patchsplit      , dest.rows, dest.cols, rows , cols );
            Bayersplit2 << <gridSize, blockSize >> > (dev_dest, dev_destinationsplit, dest.rows, dest.cols, rows , cols );break;}
    case 4:{Bayersplit4 << <gridSize, blockSize >> > (dev_src , dev_patchsplit      , dest.rows, dest.cols, rows , cols );
            Bayersplit4 << <gridSize, blockSize >> > (dev_dest, dev_destinationsplit, dest.rows, dest.cols, rows , cols );break;}
    default:{cout<<"The parameter bayer should be 1 or 2 or 4 !"<<endl;break;}
    }
//---------------------------------------------------------------------
    divergance << <gridSize, blockSize >> > (dev_destinationsplit ,dev_patchsplit,dev_binaryMaskFloatInverted , dev_binaryMaskFloat,
                                             laplacianX,laplacianY,rows, cols);
    laplacian_add << <gridSize, blockSize >> > (laplacianX, laplacianY, dev_lap,rows, cols);
    gpu_rectangle << <gridSize, blockSize >> > (dev_destinationsplit, dev_destination, rows, cols);
    gpu_Laplacian << <gridSize, blockSize >> > (dev_destination, dev_boundary_points, rows, cols);
    gpu_mod_diff << <gridSize, blockSize >> > (dev_lap, dev_boundary_points, mod_diff, rows, cols);
//----------------------------------------------------------------------------------
        dim3 gridSize2((int)ceil((float)(rows - 2) / (float)blockSize.x), (int)ceil((float)(2 * (cols - 2) + 2) / (float)blockSize.y));
        gpu_mod_rowtemp << <gridSize2, blockSize >> > (mod_diff,
                                                       mod_rowtemp[0], mod_rowtemp[1], mod_rowtemp[2],
                                                       (rows - 2), (2 * (cols - 2) + 2), (cols - 2));

        cufftExecC2C(plan, mod_rowtemp[0], mod_rowtemp[0], CUFFT_FORWARD);
        cufftExecC2C(plan, mod_rowtemp[1], mod_rowtemp[1], CUFFT_FORWARD);
        cufftExecC2C(plan, mod_rowtemp[2], mod_rowtemp[2], CUFFT_FORWARD);

        dim3 gridSize3((int)ceil((float)(cols - 2) / (float)blockSize.x), (int)ceil((float)(2 * (rows - 2) + 2) / (float)blockSize.y));
        gpu_mod_coltemp << <gridSize3, blockSize >> > (mod_rowtemp[0], mod_rowtemp[1], mod_rowtemp[2],
                                                       mod_coltemp[0], mod_coltemp[1], mod_coltemp[2],
                                                       (cols- 2), (2 * (rows - 2) + 2), (2 * (cols - 2) + 2), (rows - 2));

        cufftExecC2C(plan2, mod_coltemp[0], mod_coltemp[0], CUFFT_FORWARD);
        cufftExecC2C(plan2, mod_coltemp[1], mod_coltemp[1], CUFFT_FORWARD);
        cufftExecC2C(plan2, mod_coltemp[2], mod_coltemp[2], CUFFT_FORWARD);

        dim3 gridSize4((int)ceil((float)(rows - 2) / (float)blockSize.x), (int)ceil((float)(cols - 2) / (float)blockSize.y));
        gpu_mod_ROI << <gridSize4, blockSize >> > (mod_coltemp[0], mod_coltemp[1], mod_coltemp[2],
                                                   mod_ROI, dev_filter_X, dev_filter_Y,
                                                   (rows - 2), (cols - 2), (2 * (rows - 2) + 2));

        gpu_mod_rowtemp << <gridSize2, blockSize >> > (mod_ROI,
                                                       mod_rowtemp[0], mod_rowtemp[1], mod_rowtemp[2],
                                                       (rows - 2), (2 * (cols - 2) + 2), (cols - 2));

        cufftExecC2C(plan, mod_rowtemp[0], mod_rowtemp[0], CUFFT_FORWARD);
        cufftExecC2C(plan, mod_rowtemp[1], mod_rowtemp[1], CUFFT_FORWARD);
        cufftExecC2C(plan, mod_rowtemp[2], mod_rowtemp[2], CUFFT_FORWARD);

        gpu_mod_coltemp2 << <gridSize3, blockSize >> > (mod_rowtemp[0], mod_rowtemp[1], mod_rowtemp[2],
                mod_coltemp[0], mod_coltemp[1], mod_coltemp[2],
                (cols - 2), (2 * (rows - 2) + 2), (2 * (cols - 2) + 2), (rows - 2));

        cufftExecC2C(plan2, mod_coltemp[0], mod_coltemp[0], CUFFT_FORWARD);
        cufftExecC2C(plan2, mod_coltemp[1], mod_coltemp[1], CUFFT_FORWARD);
        cufftExecC2C(plan2, mod_coltemp[2], mod_coltemp[2], CUFFT_FORWARD);

        gpu_mod_ROI3 << <gridSize4, blockSize >> > (mod_coltemp[0], mod_coltemp[1], mod_coltemp[2],
                mod_ROI, (rows - 2), (cols - 2), (2 * (rows - 2) + 2));

        gpu_mod_ROI4 << <gridSize4, blockSize >> > (mod_ROI, dev_destinationsplit, (rows - 2), (cols - 2), (2 * (rows - 2) + 2));
        switch(bayer)
        {
        case 1:{Bayer_merge1<< <gridSize, blockSize >> > (dev_dest, dev_destinationsplit, dest.rows, dest.cols, rows , cols );break;}
        case 2:{Bayer_merge2<< <gridSize, blockSize >> > (dev_dest, dev_destinationsplit, dest.rows, dest.cols, rows , cols );break;}
        case 4:{Bayer_merge4<< <gridSize, blockSize >> > (dev_dest, dev_destinationsplit, dest.rows, dest.cols, rows , cols );break;}
        default:{cout<<"Error in Bayer!"<<endl;break;}
        }

//------------------------split ori------------------------------------
//        cudaMemcpy(dest.data, dev_dest, sizeof(uchar) *(dest.cols)*(dest.rows), cudaMemcpyDeviceToHost);
//        dest.copyTo(dest_channel[1](roi_d));
//        dest_channel[1](roi_d)=dest;
//        cvtColor(dest_channel[1], result, COLOR_BayerBG2BGR);
//------------------------split dst------------------------------------
        cudaMemcpy(dest_channel[0].data, dev_dest, sizeof(uchar) *(dest.cols)*(dest.rows), cudaMemcpyDeviceToHost);
        Mat result_channel[3];
        split(dest_ori,result_channel);
        dest_channel[0].copyTo(result_channel[0](roi_d));
        cvtColor(result_channel[0], result, COLOR_BayerBG2BGR);
//---------------------------------------------------------------------
//        out.result=result;

//       time_end=(double)clock();
//       time_diff =(time_end-time_end)/1000.0;
//       printf("runtime:  %3.0f ms\n", time_diff);

//       cudaFree( dev_destination );
//       cudaFree(destinationGradientX);
//       cudaFree(destinationGradientY);
//       cudaFree(dev_patch);
//       cudaFree(patchGradientX);
//       cudaFree(patchGradientY);
//       cudaFree(dev_binaryMaskFloat);
//       cudaFree(dev_binaryMaskFloatInverted);
//       cudaFree(laplacianX);
//       cudaFree(laplacianY);
//       cudaFree(dev_laplacianX);
//       cudaFree(dev_laplacianY);
//       cudaFree(dev_lap);
//       cudaFree(dev_boundary_points);
//       cudaFree(mod_diff);
//       for (int ii = 0; ii < 3; ii++)
//       {
//           cudaFree(mod_rowtemp[ii]);
//           cudaFree(mod_coltemp[ii]);
//       }
//       cudaFree(mod_ROI);
//       cudaFree(dev_filter_X);
//       cudaFree(dev_filter_Y);

//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&elapsedTime, start, stop);
//        Time +=elapsedTime;
//-------------------------------------------------------------------------------------
//      cudaEventRecord(stop, 0);
//      cudaEventSynchronize(stop);
//      cudaEventElapsedTime(&elapsedTime, start, stop);
//      Time +=elapsedTime;
//      printf("cuda time:  %3.0f ms\n", elapsedTime);
//      cout<<"In total: "<<Time<<" ms"<<endl;
//    imwrite("/home/nvidia/Desktop/Camerafuse/bayer_splitdst.bmp", result);
//       dest(roi_s)=result(roi_s);
//       dest_ori(roi_d)=dest(roi_d);
//    namedWindow("result",0);
//    resizeWindow("result",960,600);
//    imshow("result", result);
//    waitKey(0);
       return result;
}
//--------------------------------------------------------------------------------------
//    Mat destX = Mat::zeros(rows - 2, 2 * (cols - 2) + 2 , CV_32FC1);
//    Mat destY = Mat::zeros(rows - 2, 2 * (cols - 2) + 2 , CV_32FC1);
//    cudaMemcpy(destX.data,&mod_rowtemp[0]->x,
//               sizeof(float) *(rows - 2)*(2 * (cols - 2) + 2),cudaMemcpyDeviceToHost);
//    namedWindow("mod_diff",0);
//    resizeWindow("mod_diff",960,600);
//    imshow("mod_diff", destX);
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
//    cudaMemcpy(destY.data,&mod_rowtemp[0]->x,
//               sizeof(float) *(rows - 2)*(2 * (cols - 2) + 2),cudaMemcpyDeviceToHost);
//    namedWindow("mod_diff2",0);
//    resizeWindow("mod_diff2",960,600);
//    imshow("mod_diff2", destY);
//    /*----------------------------------*/waitKey(0);/*-------------------------------*/
