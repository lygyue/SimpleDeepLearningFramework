/***********************************************
 * File: SdlfFunction.cpp
 *
 * Author: LYG
 * Date: 十一月 2020
 *
 * Purpose:定义常用函数
 *
 * 
 **********************************************/
#include <math.h>
#include <algorithm>
#include "SdlfFunction.h"
#include "Common.h"

bool SdlfFunction::RealEqual(float a, float b, float tolerance /* = 1e-4 */)
{
	return fabs(a - b) < tolerance;
}

float SdlfFunction::MaxInPool(float* Data, int Num, int& MaxIndex)
{
	// 简单冒泡一轮
	float MaxValue = Data[0];
	MaxIndex = 0;
	for (int i = 1; i < Num; i++)
	{
		if (MaxValue < Data[i])
		{
			MaxValue = Data[i];
			MaxIndex = i;
		}
	}

	return MaxValue;
}

float SdlfFunction::AverageInPool(float* Data, int Num)
{
	float Sum = 0.0f;
	for (int i = 0; i < Num; i++)
	{
		Sum += Data[i];
	}
	return Sum / float(Num);
}

float SdlfFunction::DotProduct(float* Data1, float* Datae2, int Num)
{
	float Sum = 0.0f;
	for (int i = 0; i < Num; i++)
	{
		Sum += Data1[i] * Datae2[i];
	}

	return Sum;
}

double SdlfFunction::Pow(double x, double y)
{
	return pow(x, y);
}

double SdlfFunction::Exp(double n)
{
	return exp(n);
}

double SdlfFunction::ln(double n)
{
	return log(n);
}

double SdlfFunction::lg(double n)
{
	return log10(n);
}

double SdlfFunction::log_m_n(double m, double n)
{
	return ln(n) / ln(m);
}

double SdlfFunction::sigmoid(double z)
{
	return 1.0f / (1.0f + Exp(-z));
}

void SdlfFunction::softmax(float* InArrayDest, float* OutArray, int Num)
{
	// 增加这一步主要是解决数据可能的溢出问题。例如e的负无穷次方，会得到结果0.有可能造成除数为0的情况
	float alpha = *std::max_element(InArrayDest, InArrayDest + Num);

	double sum = 0;
	double* out_Array = new double[Num];
	for (int i = 0; i < Num; i++)
	{
		out_Array[i] = Exp(InArrayDest[i] - alpha);
		sum += out_Array[i];
	}

	for (int i = 0; i < Num; i++)
	{
		OutArray[i] = float(out_Array[i] / sum);
	}
	delete[]out_Array;
	return;
}

void SdlfFunction::softmax_default(float* InArrayDest, float* OutArray, int Num)
{
	float sum = 0;

	for (int i = 0; i < Num; i++)
	{
		OutArray[i] = (float)Exp(InArrayDest[i]);
		sum += OutArray[i];
	}

	for (int i = 0; i < Num; i++)
	{
		OutArray[i] /= sum;
	}

	return;
}

double SdlfFunction::CrossEntropy(float* InArray, int* InArrayDest, int Num)
{
	for (int i = 0; i < Num; i++)
	{
		if (InArrayDest[i] == 1)
		{
			return -ln(InArray[i]);
		}
	}
	return 0.0f;
}

float SdlfFunction::UnitRandom()
{
	static bool firstRun = true;
	if (firstRun)
	{
		srand((unsigned)time(NULL));
		firstRun = false;
	}
	return float(rand()) / float(RAND_MAX);
}

float SdlfFunction::RangeRandom(float Min, float Max)
{
	return (Max - Min)*UnitRandom() + Min;
}

void SdlfFunction::DropOut(float& Data, float level)
{
	// 这里不做level的0-1判定，浪费时间
	float remain = 1.0f - level;

	float r = RangeRandom(0.0f, 1.0f);
	if (r < level)
	{
		Data = 0.0f;		// 抛弃
	}
	else
	{
		// 为什么要干这个？我没找到合理的解释。但是看到官方的代码都有这个，加上。
		Data /= remain;
	}
	return;
}

void SdlfFunction::DropOut(float* Data, float* GradientData, int Num, float level)
{
	// 这里不做level的0-1判定，浪费时间
	float remain = 1.0f - level;

	for (int i = 0; i < Num; i++)
	{
		float r = RangeRandom(0.0f, 1.0f);
		if (r < level)
		{
			Data[i] = 0.0f;		// 抛弃
			GradientData[i] = 0.0f;
		}
		else
		{
			// 为什么要干这个？我没找到合理的解释。但是看到官方的代码都有这个，加上。
			Data[i] /= remain;
			GradientData[i] = 1 / remain;					// 这两个其实没看懂，我觉得直接写1估计也没什么问题。但是看官网都是这样写。先出来结果再说。
		}
	}
	return;
}

void SdlfFunction::BuildConvArrayPerBatch(struct ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchCountIndex)
{
	// 每个小块的大小，是卷积核的大小
	int ConvLen = ConvWidth * ConvHeight * ImageDepth;
	// 遍历图片，一块一块的生成小块
	// 每个Batch有这么多个小块
	int ImageSizePerChannel = ImageHeight * ImageWidth;
	int HalfConv = ConvWidth >> 1;
	float* ConvQuad = new float[ConvLen];
	for (int i = 0; i < ImageHeight; i++)
	{
		for (int j = 0; j < ImageWidth; j++)
		{
			// 开始填充quad
			for (int k = 0; k < ImageDepth; k++)
			{
				for (int m = 0; m < ConvHeight; m++)
				{
					for (int n = 0; n < ConvWidth; n++)
					{
						int ConvPos = k * ConvHeight * ConvWidth + m * ConvWidth + n;
						int x_ = n - HalfConv;
						int y_ = m - HalfConv;
						int x = j + x_;
						int y = i + y_;

						if (x < 0 || x >= ImageWidth || y < 0 || y >= ImageHeight)
						{
							ConvQuad[ConvPos] = 0;
						}
						else
						{
							int Pos = ImageSizePerChannel * k + y * ImageWidth + x;
							ConvQuad[ConvPos] = ImageData[Pos];
						}
					}
				}
			}
			memcpy(CB->ConvArray + (ImageSizePerChannel * BatchCountIndex + i * ImageWidth + j) * ConvLen, ConvQuad, sizeof(float) * ConvLen);
		}
	}
	SAFE_DELETE_ARRAY(ConvQuad);
}

void SdlfFunction::BuildConvArray(struct ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchCount)
{
	int ConvLen = ConvWidth * ConvHeight * ImageDepth;
	int ImageSize = ImageWidth * ImageHeight * ImageDepth;
	if (CB->ConvArray == nullptr)
	{
		CB->ConvArray = new float[ImageWidth * ImageHeight * BatchCount * ConvLen];
		CB->ArrayLen = ConvLen;
	}
	
	for (int i = 0; i < BatchCount; i++)
	{
		int Pos = i * ImageSize;
		BuildConvArrayPerBatch(CB, &ImageData[Pos], ImageWidth, ImageHeight, ImageDepth, ConvWidth, ConvHeight, i);
	}

	return;
}

void SdlfFunction::BuildFullConnectedArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchIndex)
{
	int ImageSize = ImageWidth * ImageHeight * ImageDepth;

	memcpy(CB->ConvArray + BatchIndex * ImageSize, ImageData, sizeof(float) * ImageSize);
}

void SdlfFunction::BuildFullConnectedArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchCount)
{
	int ImageSize = ImageWidth * ImageHeight * ImageDepth;
	if (CB->ConvArray == nullptr)
	{
		CB->ArrayLen = ImageSize;
		CB->ConvArray = new float[ImageSize * BatchCount];
	}

	for (int i = 0; i < BatchCount; i++)
	{
		BuildFullConnectedArrayPerBatch(CB, ImageData + i * ImageSize, ImageWidth, ImageHeight, ImageDepth, i);
	}
}

void SdlfFunction::Conv2DPerImagePerConvKernel(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, int ConvKernelIndex, float* ImageOutData, float* ReluOutData)
{
	int StartPos = ConvBlocksPerBatch * BatchIndex;
	for (int i = 0; i < ConvBlocksPerBatch; i++)
	{
		float ConvResult = CK->Conv2D(&CB->ConvArray[(StartPos + i) * CB->ArrayLen], ConvKernelIndex);
		if (ConvResult > 0)
		{
			// 假定ImageOutData和ReluOutData已经初始化为0
			ImageOutData[i] = ConvResult;
			ReluOutData[i] = 1.0f;
		}
	}
}
void SdlfFunction::Conv2DPerImage(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData)
{
	for (int i = 0; i < CK->ConvKernelCount; i++)
	{
		Conv2DPerImagePerConvKernel(CB, ConvBlocksPerBatch, BatchIndex, CK, i, &ImageOutData[ImageSize2D * i], &ReluOutData[ImageSize2D * i]);
	}
}
void SdlfFunction::Conv2D(ConvBlocks* CB, int ConvBlocksPerBatch, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData, int BatchCount)
{
	int ImageSize3D = ImageSize2D * CK->ConvKernelCount;
	for (int i = 0; i < BatchCount; i++)
	{
		Conv2DPerImage(CB, ConvBlocksPerBatch, i, CK, &ImageOutData[i * ImageSize3D], ImageSize2D, &ReluOutData[i * ImageSize3D]);
	}
}

void SdlfFunction::ConvFullConnected(ConvBlocks* CB, ConvKernel* CK, float* ImageOutData, float* ReluOutData, int BatchCount)
{
	// 对于全连接来说，一张图片生成一个卷积，所以这里的卷积数，就是batch数
	for (int i = 0; i < BatchCount; i++)
	{
		for (int j = 0; j < CK->ConvKernelCount; j++)
		{
			float ConvResult = CK->Conv2D(&CB->ConvArray[i * CB->ArrayLen], j);
			if (ConvResult > 0.0f)
			{
				ImageOutData[i * CK->ConvKernelCount + j] = ConvResult;
				ReluOutData[i * CK->ConvKernelCount + j] = 1.0f;
			}
		}
	}
}