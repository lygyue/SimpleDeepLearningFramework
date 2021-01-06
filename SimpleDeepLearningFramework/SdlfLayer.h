/***********************************************
 * File: SdlfLayer.h
 *
 * Author: LYG
 * Date: 十一月 2020
 *
 * Purpose:
 *
 * 
 **********************************************/
#pragma once

#include "Common.h"
#include "Sdlf.h"
#include "SdlfCalculator.h"

class SdlfLayerImpl : public SdlfLayer
{
public:
	SdlfLayerImpl();
	virtual ~SdlfLayerImpl();


	virtual void SetLayerType(SdlfLayerType LT) override;
	virtual SdlfLayerType GetLayerType() const override;

	virtual SdlfLayer* GetPreLayer() override;
	virtual void SetPreLayer(SdlfLayer* Layer)override;
	virtual SdlfLayer* GetNextLayer() override;
	virtual void SetNextLayer(SdlfLayer* Layer) override;

	// Must be convolution layer
	virtual bool SetConvolutionKernel(int ConvKernelWidth, int ConvKernelHeight, int InWidth, int InHeight, int InChannels, int OutChannels) override;
	// 初始化卷积核的参数。这个默认是随机的。但是有需要的话，我看也有初始化的？
	// 只设置w跟b
	virtual void SetConvolutionParam(float W, float B) override;
	virtual void InitialiseConvParamToRamdom(float fMin, float fMax) override;
	virtual bool SetActivationFunc(SdlfActivationFunc AF) override;
	virtual bool SetPoolParameter(PoolType PT) override;
	// 全连接层参数设置。据说现在很多的算法已经抛弃了全连接层，而用的全局求平均值。
	// 这里也考虑一下，做一个均值的测试一下效果，看看有没有什么不同
	virtual bool SetFullLinkParameter(int Width, int Height, int Depth, int FullLinkDepth) override;

	// 设置求导参数，目前就设置一个step先
	virtual void SetGradientParameter(float Step) override;
	virtual void Release() override;

	float* Excute(float* ImageData, int BatchCount, SdlfCalculator* Calculator);
	// 从Calculator获取上一次的输出。因为这次的输出可能是在GPU端，完全没必要读取出来。
	// 返回分类输出结果，可以逆向求导或者是计算准确率
	float* Excute(SdlfCalculator* Calculator);
	// 反向求导
	void SoftMaxGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator);
	void FullLinkGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator);
	void Conv2DGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator);

	void UpdateStep(float Multi);
protected:
	ConvKernel* GetConvKernel();
	ConvKernel* GetFullLinkKernel();
private:
	SdlfLayerType mLayerType;
	SdlfActivationFunc mActivationFunc;
	PoolType mPoolType;
	// 记录一些参数。还需要计算一下padding，我们假设自动计算这个，让每一次卷积都不会改变宽高。我们假设图片宽高一致
	int mInWidth;
	int mInHeight;
	int mInChannel;
	int mPadding;

	SdlfLayerImpl* mPreLayer;
	SdlfLayerImpl* mNextLayer;

	float mStep;
	// 卷积核
	ConvKernel* mConvKernel;

	// 全连接参数，预计也是需要求导更新的
	ConvKernel* mFullLinkKernel;
	// SoftMax参数
	SoftMaxKernel* mSoftMaxKernel;
};