/***********************************************
 * File: SdlfCalculatorCPU.h
 *
 * Author: LYG
 * Date: ʮһ�� 2020
 *
 * Purpose:
 *
 * 
 **********************************************/
#pragma once

#include "SdlfCalculator.h"

// ������CPU���㣬�Ͳ�������SIMDָ�����Ż��ˡ���Ϊ��ô�Ż���ƴ����GPU��CPU����������ô��뿴��������һЩ���õ���һЩ��
class SdlfCalculatorCPU : public SdlfCalculator
{
	friend class SdlfCalculator;
public:
	virtual void Conv2D(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchCount, ConvKernel* CK, SdlfActivationFunc ActivationFunc) final;
	virtual void Conv2D(ConvKernel* CK, SdlfActivationFunc ActivationFunc) final;
	virtual void Conv3D() final;

	// ��ȡ�ϴ��������Ϊ�ϴ�������������Դ���ģ����ʱ����ȫ����Ҫ���¶����ڴ棬ֱ�ӻ�ȡһ��ָ�뼴��
	virtual void* GetLastOutput(int& Width, int& Height, int& Depth, int& BatchCount) final;

	// ͼƬ��ʽת����һ��ͼƬ����uchar������������������������float��Ĭ��ת��float��-1,1֮��
	virtual void Transform_uchar_to_float(unsigned char* InData, float* OutData, int W, int H, int Channel) final;
	// ͼƬ��ʽת�������ǵ�batch��������GPUת�������ʱ�����õ�һ��ͼƬ����
	virtual void Transform_uchar_to_float(unsigned char* InData, float* OutData, int DataLen) final;
	// ת��float��0��1֮�䣬��ͬ
	virtual void Transform_uchar_to_float_0_1(unsigned char* InData, float* OutData, int W, int H, int Channel) final;
	virtual void Transform_uchar_to_float_0_1(unsigned char* InData, float* OutData, int DataLen) final;
	virtual void Transform_float_to_uchar(float* InData, unsigned char* OutData, int W, int H, int Channel) final;
	virtual void Transform_float_to_uchar(float* InData, unsigned char* OutData, int DataLen) final;
	virtual void Transform_float_0_1_to_uchar(float* InData, unsigned char* OutData, int W, int H, int Channel) final;
	virtual void Transform_float_0_1_to_uchar(float* InData, unsigned char* OutData, int DataLen) final;

	// �ػ�����
	virtual void Max_Pool_2_2(int InWidth, int InHeight, float* InData, float* MaxPoolGradientData, float* OutData) final;
	virtual void Max_Pool_2_2(ConvKernel* CK) final;
	virtual void Average_Pool_2_2(int InWidth, int InHeight, int InChannel, float* InData, float* OutData) final;
	virtual void Average_Pool_2_2(ConvKernel* CK) final;

	// ȫ���Ӳ����
	virtual void CalcFullLink(ConvKernel* FLinkKernel) final;
	virtual void CalcFullLink(float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, ConvKernel* FLinkKernel) final;
	// softmax
	virtual float* SoftMax(SoftMaxKernel* SMK) final;

	virtual void Release() final;
protected:
	SdlfCalculatorCPU();
	virtual ~SdlfCalculatorCPU();
private:
	// ͼƬ�����Ϣ��
	float* mImageOutData;
};