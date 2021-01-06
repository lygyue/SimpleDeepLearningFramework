/***********************************************
 * File: SdlfCalculatorCPU.h
 *
 * Author: LYG
 * Date: 十一月 2020
 *
 * Purpose:
 *
 * 
 **********************************************/
#pragma once

#include "SdlfCalculator.h"

// 这里用CPU计算，就不打算用SIMD指令做优化了。因为怎么优化都拼不过GPU，CPU的意义就是让代码看起来容易一些，好调试一些。
class SdlfCalculatorCPU : public SdlfCalculator
{
	friend class SdlfCalculator;
public:
	virtual void Conv2D(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchCount, ConvKernel* CK, SdlfActivationFunc ActivationFunc) final;
	virtual void Conv2D(ConvKernel* CK, SdlfActivationFunc ActivationFunc) final;
	virtual void Conv3D() final;

	// 获取上次输出。因为上次输出可能是在显存里的，这个时候完全不需要重新读出内存，直接获取一个指针即可
	virtual void* GetLastOutput(int& Width, int& Height, int& Depth, int& BatchCount) final;

	// 图片格式转换。一般图片都是uchar，但是真正计算力基本都是float，默认转换float是-1,1之间
	virtual void Transform_uchar_to_float(unsigned char* InData, float* OutData, int W, int H, int Channel) final;
	// 图片格式转换，考虑到batch，可能用GPU转换，这个时候不能用单一的图片处理
	virtual void Transform_uchar_to_float(unsigned char* InData, float* OutData, int DataLen) final;
	// 转换float是0，1之间，下同
	virtual void Transform_uchar_to_float_0_1(unsigned char* InData, float* OutData, int W, int H, int Channel) final;
	virtual void Transform_uchar_to_float_0_1(unsigned char* InData, float* OutData, int DataLen) final;
	virtual void Transform_float_to_uchar(float* InData, unsigned char* OutData, int W, int H, int Channel) final;
	virtual void Transform_float_to_uchar(float* InData, unsigned char* OutData, int DataLen) final;
	virtual void Transform_float_0_1_to_uchar(float* InData, unsigned char* OutData, int W, int H, int Channel) final;
	virtual void Transform_float_0_1_to_uchar(float* InData, unsigned char* OutData, int DataLen) final;

	// 池化计算
	virtual void Max_Pool_2_2(int InWidth, int InHeight, float* InData, float* MaxPoolGradientData, float* OutData) final;
	virtual void Max_Pool_2_2(ConvKernel* CK) final;
	virtual void Average_Pool_2_2(int InWidth, int InHeight, int InChannel, float* InData, float* OutData) final;
	virtual void Average_Pool_2_2(ConvKernel* CK) final;

	// 全连接层计算
	virtual void CalcFullLink(ConvKernel* FLinkKernel) final;
	virtual void CalcFullLink(float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, ConvKernel* FLinkKernel) final;
	// softmax
	virtual float* SoftMax(SoftMaxKernel* SMK) final;

	virtual void Release() final;
protected:
	SdlfCalculatorCPU();
	virtual ~SdlfCalculatorCPU();
private:
	// 图片输出信息。
	float* mImageOutData;
};