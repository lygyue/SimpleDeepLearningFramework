/***********************************************
 * File: SdlfCalculator.h
 *
 * Author: LYG
 * Date: 十一月 2020
 *
 * Purpose:把大量的计算集中在一个地方，做一个抽象
 * 底层可以是CPU，Compute Shader, Cuda, Cudnn……暂时只打算实现CPU跟Compute Shader。
 * 理论上，CPU效率最低，但是能容易看清楚算法。Compute Shader，Cuda很相似，最大的区别貌似是浮点数精度不同。但是CS不挑显卡，什么显卡都行，Cuda必须N卡。
 * Cudnn最简单，因为Cudnn本身应该集成了一大堆现成算法，直接调用即可。我希望自己实现一遍底层算法，加深自己的理解。
 **********************************************/

#pragma once
#include "Common.h"

enum SdlfCalculatorType
{
	SCT_CPU,
	SCT_ComputeShader,
	SCT_Cuda,
	SCT_Cudnn,
};
enum SdlfActivationFunc;
class SdlfCalculator
{
public:
	static SdlfCalculator* CreateCalculator(SdlfCalculatorType SCT);

	virtual void Conv2D(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchCount, ConvKernel* CK, SdlfActivationFunc ActivationFunc) = 0;
	virtual void Conv2D(ConvKernel* CK, SdlfActivationFunc ActivationFunc) = 0;
	virtual void Conv3D() = 0;

	// 获取上次输出。因为上次输出可能是在显存里的，这个时候完全不需要重新读出内存，直接获取一个指针即可
	virtual void* GetLastOutput(int& Width, int& Height, int& Depth, int& BatchCount) = 0;

	// 图片格式转换。一般图片都是uchar，但是真正计算里基本都是float，默认转换float是-1,1之间
	virtual void Transform_uchar_to_float(unsigned char* InData, float* OutData, int W, int H, int Channel) = 0;
	// 图片格式转换，考虑到batch，可能用GPU转换，这个时候不能用单一的图片处理
	virtual void Transform_uchar_to_float(unsigned char* InData, float* OutData, int DataLen) = 0;
	// 转换float是0，1之间，下同
	virtual void Transform_uchar_to_float_0_1(unsigned char* InData, float* OutData, int W, int H, int Channel) = 0;
	virtual void Transform_uchar_to_float_0_1(unsigned char* InData, float* OutData, int DataLen) = 0;
	virtual void Transform_float_to_uchar(float* InData, unsigned char* OutData, int W, int H, int Channel) = 0;
	virtual void Transform_float_to_uchar(float* InData, unsigned char* OutData, int DataLen) = 0;
	virtual void Transform_float_0_1_to_uchar(float* InData, unsigned char* OutData, int W, int H, int Channel) = 0;
	virtual void Transform_float_0_1_to_uchar(float* InData, unsigned char* OutData, int DataLen) = 0;

	// 池化计算，第二个函数是已知其他参数。例如是GPU实现，底层直接实现，不需要传入传出参数
	virtual void Max_Pool_2_2(int InWidth, int InHeight, float* InData, float* MaxPoolGradientData, float* OutData) = 0;
	virtual void Max_Pool_2_2(ConvKernel* CK) = 0;
	virtual void Average_Pool_2_2(int InWidth, int InHeight, int InChannel, float* InData, float* OutData) = 0;
	virtual void Average_Pool_2_2(ConvKernel* CK) = 0;

	// 全连接层计算
	virtual void CalcFullLink(ConvKernel* FLinkKernel) = 0;
	virtual void CalcFullLink(float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, ConvKernel* FLinkKernel) = 0;

	// softmax
	virtual float* SoftMax(SoftMaxKernel* SMK) = 0;

	virtual void Release() = 0;
protected:
	SdlfCalculator() 
	{
		mInWidth			= 0;
		mInHeight			= 0;
		mInChannel			= 0;
		mOutWidth			= 0;
		mOutHeight			= 0;
		mOutChannel			= 0;
		mPadding			= 0;
		mBatchCount			= 32;
	};
	virtual ~SdlfCalculator() {};

	static float UCharToFloat(unsigned char C);
	static float UCharToFloat_0_1(unsigned char C);
	static unsigned char FloatToUChar(float F);
	static unsigned char Float_0_1_ToUChar(float F);


	int mInWidth;
	int mInHeight;
	int mInChannel;
	int mOutWidth;
	int mOutHeight;
	int mOutChannel;

	int mPadding;
	int mBatchCount;
};

