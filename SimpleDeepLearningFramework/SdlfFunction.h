/***********************************************
 * File: SdlfFunction.h
 *
 * Author: LYG
 * Date: 十一月 2020
 *
 * Purpose:
 *
 * 
 **********************************************/
#pragma once
#include <time.h>

struct  ConvBlocks;
struct ConvKernel;
// 把常用函数这里再封装一次，主要有两个原因：
// 1、有些函数需要常用函数多次计算；2、有时候写代码会忘记一些常用函数，还要查资料，还不如自己封装一次，要查的时候这里找。
class SdlfFunction
{
public:
	static bool RealEqual(float a, float b, float tolerance = 1e-4);

	// 求最大值，主要是池化会用到？
	static float MaxInPool(float* Data, int Num, int& MaxIndex);
	// 求平均值
	static float AverageInPool(float* Data, int Num);
	// 点积
	static float DotProduct(float* Data1, float* Datae2, int Num);
	// x^y
	static double Pow(double x, double y);
	// e^n
	static double Exp(double n);
	// ln(n)，e为底数的对数函数
	static double ln(double n);
	// lg(n)，10为底数的对数函数
	static double lg(double n);
	// log(m,n)，m为底数的对数函数。有时候要用到这种，例如交叉熵，有用e为底数，但是我看有些地方是2为底数
	static double log_m_n(double m, double n);
	// sigmoid，非常常用的函数，无论是激活函数还是二分类的时候替换softmax
	static double sigmoid(double z);
	// softmax，分类问题里面非常常用的函数
	// 这里用float作为输入，是因为使用GPU计算的时候，全部都是float。这里使用double没什么意义。
	// InArray和OutArray必须有一样的Num
	// InArrayDest：目标结果，InArrayDest：实际计算结果
	// 这里做了一些特殊处理，避免了除数为0的情况。
	static void softmax(float* InArrayDest, float* OutArray, int Num);
	// 默认的softmax函数，不做特殊处理
	static void softmax_default(float* InArrayDest, float* OutArray, int Num);
	// CrossEntropy，交叉熵。
	static double CrossEntropy(float* InArray, int* InArrayDest, int Num);

	// 单位随机数
	static float UnitRandom();
	//范围随机数
	static float RangeRandom(float Min, float Max);
	// dropout
	// level是损失概率值，在0-1之间
	static void DropOut(float& Data, float level);
	static void DropOut(float* Data, float* GradientData, int Num, float level);

	// 这里不输入卷积核的depth，这个必须跟输入image的depth一致
	static void BuildConvArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchCountIndex);
	static void BuildConvArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchCount);
	// 这个其实很简单，但是为了跟卷积核对齐，所以用了一样的方式
	static void BuildFullConnectedArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchIndex);
	static void BuildFullConnectedArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchCount);

	// 这里有点绕，一张图片，需要跟N个卷积核做卷积，生成N张图片。这里是一张图片做卷积，结果也是一张图片。
	// 为什么绕，因为涉及到batch，涉及到多个卷积核。而batch再多，卷积核数量是不变的
	static void Conv2D(ConvBlocks* CB, int ConvBlocksPerBatch, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData, int BatchCount);
	static void Conv2DPerImagePerConvKernel(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, int ConvKernelIndex, float* ImageOutData, float* ReluOutData);
	static void Conv2DPerImage(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData);

	// 计算全连接。
	static void ConvFullConnected(ConvBlocks* CB, ConvKernel* CK, float* ImageOutData, float* ReluOutData, int BatchCount);
protected:
	SdlfFunction() {};
	~SdlfFunction() {};
};