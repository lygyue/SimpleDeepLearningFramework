/***********************************************
 * File: SdlfLayer.h
 *
 * Author: LYG
 * Date: ʮһ�� 2020
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
	// ��ʼ������˵Ĳ��������Ĭ��������ġ���������Ҫ�Ļ����ҿ�Ҳ�г�ʼ���ģ�
	// ֻ����w��b
	virtual void SetConvolutionParam(float W, float B) override;
	virtual void InitialiseConvParamToRamdom(float fMin, float fMax) override;
	virtual bool SetActivationFunc(SdlfActivationFunc AF) override;
	virtual bool SetPoolParameter(PoolType PT) override;
	// ȫ���Ӳ�������á���˵���ںܶ���㷨�Ѿ�������ȫ���Ӳ㣬���õ�ȫ����ƽ��ֵ��
	// ����Ҳ����һ�£���һ����ֵ�Ĳ���һ��Ч����������û��ʲô��ͬ
	virtual bool SetFullLinkParameter(int Width, int Height, int Depth, int FullLinkDepth) override;

	// �����󵼲�����Ŀǰ������һ��step��
	virtual void SetGradientParameter(float Step) override;
	virtual void Release() override;

	float* Excute(float* ImageData, int BatchCount, SdlfCalculator* Calculator);
	// ��Calculator��ȡ��һ�ε��������Ϊ��ε������������GPU�ˣ���ȫû��Ҫ��ȡ������
	// ���ط��������������������󵼻����Ǽ���׼ȷ��
	float* Excute(SdlfCalculator* Calculator);
	// ������
	void SoftMaxBackward_Propagation(float* GradientData, int BatchCount, SdlfCalculator* Calculator);
	void FullLinkBackward_Propagation(float* GradientData, int BatchCount, SdlfCalculator* Calculator);
	void Conv2DBackward_Propagation(float* GradientData, int BatchCount, SdlfCalculator* Calculator);

	void UpdateStep(float Multi);
protected:
	ConvKernel* GetConvKernel();
	ConvKernel* GetFullLinkKernel();
private:
	SdlfLayerType mLayerType;
	SdlfActivationFunc mActivationFunc;
	PoolType mPoolType;
	// ��¼һЩ����������Ҫ����һ��padding�����Ǽ����Զ������������ÿһ�ξ��������ı��ߡ����Ǽ���ͼƬ���һ��
	int mInWidth;
	int mInHeight;
	int mInChannel;
	int mPadding;

	SdlfLayerImpl* mPreLayer;
	SdlfLayerImpl* mNextLayer;

	float mStep;
	// �����
	ConvKernel* mConvKernel;

	// ȫ���Ӳ�����Ԥ��Ҳ����Ҫ�󵼸��µ�
	ConvKernel* mFullLinkKernel;
	// SoftMax����
	SoftMaxKernel* mSoftMaxKernel;
};