/***********************************************
 * File: SdlfCalculatorCPU.cpp
 *
 * Author: LYG
 * Date: ʮһ�� 2020
 *
 * Purpose:
 *
 * 
 **********************************************/

#include "SdlfCalculatorCPU.h"
#include "SdlfFunction.h"
#include<string.h>
#include <algorithm>

SdlfCalculatorCPU::SdlfCalculatorCPU()
{
	mImageOutData = nullptr;
}

SdlfCalculatorCPU::~SdlfCalculatorCPU()
{
	SAFE_DELETE_ARRAY(mImageOutData);
}

void SdlfCalculatorCPU::Conv2D(float* ImageData, int ImageWidth, int ImageHeight, int ImageChannel, int BatchCount, ConvKernel* CK, SdlfActivationFunc ActivationFunc)
{
	mInWidth = mOutWidth = ImageWidth;
	mInHeight = mOutHeight = ImageHeight;
	mInChannel = ImageChannel;
	// ���channel�Ǿ���˵Ĳ���
	mOutChannel = CK->ConvKernelCount;
	mBatchCount = BatchCount;

	mPadding = (CK->ConvKernelWidth - 1) >> 1;

	SAFE_DELETE_ARRAY(mImageOutData);

	// ����Ҫ����CK->ConvKernelCount��һ��ͼ��ʵ�Ƿֿ���CK->ConvKernelCount��ô��������ͼ
	mImageOutData = new float[ImageWidth * ImageHeight * BatchCount * CK->ConvKernelCount];
	float* ReluGradientData = new float[mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount];

	memset(ReluGradientData, 0, sizeof(float) * mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount);
	memset(mImageOutData, 0, sizeof(float) * mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount);
	// ��ʼ���������
	if (CK->CB == nullptr)
	{
		CK->CB = new ConvBlocks;
	}
	SdlfFunction::BuildConvArray(CK->CB, ImageData, ImageWidth, ImageHeight, ImageChannel, CK->ConvKernelWidth, CK->ConvKernelHeight, BatchCount);
	// ÿ��batch���Էֳ���ô���С��
	int ConvCountPerBatch = ImageWidth * ImageHeight;
	SdlfFunction::Conv2D(CK->CB, ConvCountPerBatch, CK, mImageOutData, ConvCountPerBatch, ReluGradientData, BatchCount);

	CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount);
	CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
}

void SdlfCalculatorCPU::Conv2D(ConvKernel* CK, SdlfActivationFunc ActivationFunc)
{
	// ����һ�ε������Ϊ��ε����룬���о������
	mInWidth = mOutWidth;
	mInHeight = mOutHeight;
	mInChannel = CK->ConvKernelChannel;

	// ���channel�Ǿ���˵�channel
	mOutChannel = CK->ConvKernelCount;
	mPadding = (CK->ConvKernelWidth - 1) >> 1;

	// ����Ҫ����CK->ConvKernelCount��һ��ͼ��ʵ�Ƿֿ���CK->ConvKernelCount��ô��������ͼ
	float* OutData = new float[mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount];
	float* ReluGradientData = new float[mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount];

	memset(OutData, 0, sizeof(float) * mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount);
	memset(ReluGradientData, 0, sizeof(float) * mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount);

	// ��ʼ���������
	if (CK->CB == nullptr)
	{
		CK->CB = new ConvBlocks;
	}
	SdlfFunction::BuildConvArray(CK->CB, mImageOutData, mInWidth, mInHeight, mInChannel, CK->ConvKernelWidth, CK->ConvKernelHeight, mBatchCount);
	// ÿ��batch���Էֳ���ô���С��
	int ConvCountPerBatch = mInWidth * mInHeight;
	SdlfFunction::Conv2D(CK->CB, ConvCountPerBatch, CK, OutData, ConvCountPerBatch, ReluGradientData, mBatchCount);

	CK->ApplyImageReluGradientData(ReluGradientData, mInWidth * mInHeight * mBatchCount * CK->ConvKernelCount);
	CK->ApplyImageInputWidthAndHeight(mInWidth, mInHeight);
	SAFE_DELETE_ARRAY(ReluGradientData);
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;
}

void SdlfCalculatorCPU::Conv3D()
{

}

void* SdlfCalculatorCPU::GetLastOutput(int& Width, int& Height, int& Depth, int& BatchCount)
{
	Width = mOutWidth;
	Height = mOutHeight;
	Depth = mOutChannel;
	BatchCount = mBatchCount;
	return mImageOutData;
}

void SdlfCalculatorCPU::Transform_uchar_to_float(unsigned char* InData, float* OutData, int W, int H, int Channel)
{
	// ���������һЩ����ֱ��ʵ��µ����ݶ���֮����ˣ�˭û�¸�������ֱ���������ġ�
	int Len = W * H * Channel;

	for (int i = 0; i < Len; i++)
	{
		OutData[i] = UCharToFloat(InData[i]);
	}
}

void SdlfCalculatorCPU::Transform_uchar_to_float(unsigned char* InData, float* OutData, int DataLen)
{
	for (int i = 0; i < DataLen; i++)
	{
		OutData[i] = UCharToFloat(InData[i]);
	}
}

void SdlfCalculatorCPU::Transform_uchar_to_float_0_1(unsigned char* InData, float* OutData, int W, int H, int Channel)
{
	int Len = W * H * Channel;

	for (int i = 0; i < Len; i++)
	{
		OutData[i] = UCharToFloat_0_1(InData[i]);
	}
}

void SdlfCalculatorCPU::Transform_uchar_to_float_0_1(unsigned char* InData, float* OutData, int DataLen)
{
	for (int i = 0; i < DataLen; i++)
	{
		OutData[i] = UCharToFloat_0_1(InData[i]);
	}
}

void SdlfCalculatorCPU::Transform_float_to_uchar(float* InData, unsigned char* OutData, int W, int H, int Channel)
{
	int Len = W * H * Channel;

	for (int i = 0; i < Len; i++)
	{
		OutData[i] = FloatToUChar(InData[i]);
	}
}

void SdlfCalculatorCPU::Transform_float_to_uchar(float* InData, unsigned char* OutData, int DataLen)
{
	for (int i = 0; i < DataLen; i++)
	{
		OutData[i] = FloatToUChar(InData[i]);
	}
}

void SdlfCalculatorCPU::Transform_float_0_1_to_uchar(float* InData, unsigned char* OutData, int W, int H, int Channel)
{
	int Len = W * H * Channel;

	for (int i = 0; i < Len; i++)
	{
		OutData[i] = Float_0_1_ToUChar(InData[i]);
	}
}

void SdlfCalculatorCPU::Transform_float_0_1_to_uchar(float* InData, unsigned char* OutData, int DataLen)
{
	for (int i = 0; i < DataLen; i++)
	{
		OutData[i] = Float_0_1_ToUChar(InData[i]);
	}
}

void SdlfCalculatorCPU::Max_Pool_2_2(int InWidth, int InHeight, float* InData, float* MaxPoolGradientData, float* OutData)
{
	float *PosOut;
	float Temp[4];
	for (int i = 0; i < InHeight / 2; i++)
	{
		for (int j = 0; j < InWidth / 2; j++)
		{
			int P[4] = { (i * 2 * InWidth + j * 2), (i * 2 * InWidth + j * 2 + 1), ((i * 2 + 1) * InWidth + j * 2) , ((i * 2 + 1) * InWidth + j * 2 + 1) };
			PosOut = &(OutData[(i * InWidth / 2 + j)]);

			Temp[0] = InData[P[0]];
			Temp[1] = InData[P[1]];
			Temp[2] = InData[P[2]];
			Temp[3] = InData[P[3]];
			// �п����ĸ�ֵ��ȣ����ʱ���þ�ֵ�ػ�
			if (Temp[0] == Temp[1] && Temp[0] == Temp[2] && Temp[0] == Temp[3])
			{
				PosOut[0] = Temp[0];
				MaxPoolGradientData[P[0]] = 0.25f;
				MaxPoolGradientData[P[1]] = 0.25f;
				MaxPoolGradientData[P[2]] = 0.25f;
				MaxPoolGradientData[P[3]] = 0.25f;
			}
			else
			{
				int MaxIndex = 0;
				PosOut[0] = SdlfFunction::MaxInPool(Temp, 4, MaxIndex);
				MaxPoolGradientData[P[MaxIndex]] = 1.0f;
			}
		}
	}
	return;
}

void SdlfCalculatorCPU::Max_Pool_2_2(ConvKernel* CK)
{
	int OriginalImageSize = mInWidth * mInHeight;
	// ����4����Ϊ�ػ�����������ǿ��һ�߳���2
	int ImageSize = OriginalImageSize >> 2;
	float* OutData = new float[ImageSize * CK->ConvKernelCount * mBatchCount];
	float* MaxPoolGradientData = new float[OriginalImageSize * CK->ConvKernelCount * mBatchCount];
	memset(MaxPoolGradientData, 0, sizeof(float) * OriginalImageSize * CK->ConvKernelCount * mBatchCount);

	for (int i = 0; i < mBatchCount * CK->ConvKernelCount; i++)
	{
		int InPos = i * OriginalImageSize;
		int OutPos = i * ImageSize;
		Max_Pool_2_2(mInWidth, mInHeight, mImageOutData + InPos, MaxPoolGradientData + InPos, OutData + OutPos);
	}

	CK->ApplyImageMaxPoolGradientData(MaxPoolGradientData, OriginalImageSize * CK->ConvKernelCount * mBatchCount);

	SAFE_DELETE_ARRAY(mImageOutData);
	SAFE_DELETE_ARRAY(MaxPoolGradientData);
	mImageOutData = OutData;
	mOutWidth = mInWidth >> 1;
	mOutHeight = mInHeight >> 1;
//	mOutChannel = CK->ConvKernelCount;
}

void SdlfCalculatorCPU::Average_Pool_2_2(int InWidth, int InHeight, int InChannel, float* InData, float* OutData)
{
	float *Pos1, *Pos2, *Pos3, *Pos4, *PosOut;
	float Temp[4];
	for (int i = 0; i < InHeight / 2; i++)
	{
		for (int j = 0; j < InWidth / 2; j++)
		{
			Pos1 = &(InData[(i * 2 * InWidth + j * 2) * InChannel]);
			Pos2 = &(InData[(i * 2 * InWidth + j * 2 + 1) * InChannel]);
			Pos3 = &(InData[((i * 2 + 1) * InWidth + j * 2) * InChannel]);
			Pos4 = &(InData[((i * 2 + 1) * InWidth + j * 2 + 1) * InChannel]);
			PosOut = &(OutData[(i * InWidth / 2 + j) * InChannel]);
			for (int k = 0; k < InChannel; k++)
			{
				Temp[0] = Pos1[k];
				Temp[1] = Pos2[k];
				Temp[2] = Pos3[k];
				Temp[3] = Pos4[k];
				PosOut[k] = SdlfFunction::AverageInPool(Temp, 4);
			}
		}
	}
	return;
}

void SdlfCalculatorCPU::Average_Pool_2_2(ConvKernel* CK)
{
	int ImageSize = mInWidth * mInHeight * mInChannel;
	float* OutData = new float[ImageSize * CK->ConvKernelChannel * mBatchCount];
	memcpy(OutData, mImageOutData, ImageSize * mBatchCount * sizeof(float));

	for (int i = 0; i < mBatchCount * CK->ConvKernelChannel; i++)
	{
		int InPos = i * ImageSize;
		int OutPos = InPos / 4;
		Average_Pool_2_2(mInWidth, mInHeight, mInChannel, mImageOutData + InPos, OutData + OutPos);
	}
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;
	mOutWidth = mInWidth >> 1;
	mOutHeight = mInHeight >> 1;
	mOutChannel = mInChannel;
}

void SdlfCalculatorCPU::CalcFullLink(float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, ConvKernel* FLinkKernel)
{
	float* OutData = new float[mBatchCount * FLinkKernel->ConvKernelCount];
	float* DropOutGradient = new float[mBatchCount * FLinkKernel->ConvKernelCount];
	float* ReluGradient = new float[mBatchCount * FLinkKernel->ConvKernelCount];

	memset(OutData, 0, sizeof(float) * mBatchCount * FLinkKernel->ConvKernelCount);
	memset(DropOutGradient, 0, sizeof(float) * mBatchCount * FLinkKernel->ConvKernelCount);
	memset(ReluGradient, 0, sizeof(float) * mBatchCount * FLinkKernel->ConvKernelCount);
	// ȫ����֮�������ͼƬ��ȸ��߶�֮��ģ�ȫ��û���ˣ�Ӧ����ֻ��һ��channel�ˡ�
	mOutWidth = ImageWidth;
	mOutHeight = ImageHeight;
	mOutChannel = ImageDepth;
	mInWidth = mOutWidth;
	mInHeight = mOutHeight;
	mInChannel = mOutChannel;

	// apply image input data
	if (FLinkKernel->CB == nullptr)
	{
		FLinkKernel->CB = new ConvBlocks;
	}
	SdlfFunction::BuildFullConnectedArray(FLinkKernel->CB, ImageData, ImageWidth, ImageHeight, ImageDepth, mBatchCount);

	SdlfFunction::ConvFullConnected(FLinkKernel->CB, FLinkKernel, OutData, ReluGradient, mBatchCount);
	FLinkKernel->ApplyImageInputWidthAndHeight(mOutWidth, mOutHeight);

	// ���⣬������ֱ��DropOut���ˡ���Ȼ��Ҫ�������ڰ���
	SdlfFunction::DropOut(OutData, DropOutGradient, mBatchCount * FLinkKernel->ConvKernelCount, DROP_OUT_PARAM);
	// �������Ǳ����1����0�ˣ�һ��ͼƬ���Ѿ�ȫ�������N����
	mOutWidth = mOutHeight = 1;
	mOutChannel = FLinkKernel->ConvKernelCount;
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;

	FLinkKernel->ApplyDropOutGradient(DropOutGradient, mBatchCount * FLinkKernel->ConvKernelCount);
	SAFE_DELETE_ARRAY(DropOutGradient);
	FLinkKernel->ApplyImageReluGradientData(ReluGradient, mBatchCount * FLinkKernel->ConvKernelCount);
	SAFE_DELETE_ARRAY(ReluGradient);
}

void SdlfCalculatorCPU::CalcFullLink(ConvKernel* FLinkKernel)
{
	float* OutData = new float[mBatchCount * FLinkKernel->ConvKernelCount];
	float* DropOutGradient = new float[mBatchCount * FLinkKernel->ConvKernelCount];
	float* ReluGradient = new float[mBatchCount * FLinkKernel->ConvKernelCount];

	memset(OutData, 0, sizeof(float) * mBatchCount * FLinkKernel->ConvKernelCount);
	memset(DropOutGradient, 0, sizeof(float) * mBatchCount * FLinkKernel->ConvKernelCount);
	memset(ReluGradient, 0, sizeof(float) * mBatchCount * FLinkKernel->ConvKernelCount);

	// ȫ����֮�������ͼƬ��ȸ��߶�֮��ģ�ȫ��û���ˣ�Ӧ����ֻ��һ��channel�ˡ�
	mInWidth = mOutWidth;
	mInHeight =  mOutHeight;
	mInChannel = mOutChannel;

	// apply image input data
	if (FLinkKernel->CB == nullptr)
	{
		FLinkKernel->CB = new ConvBlocks;
	}
	SdlfFunction::BuildFullConnectedArray(FLinkKernel->CB, mImageOutData, mInWidth, mInHeight, mInChannel, mBatchCount);

	SdlfFunction::ConvFullConnected(FLinkKernel->CB, FLinkKernel, OutData, ReluGradient, mBatchCount);
	FLinkKernel->ApplyImageInputWidthAndHeight(mOutWidth, mOutHeight);

	// ���⣬������ֱ��DropOut���ˡ���Ȼ��Ҫ�������ڰ���
	SdlfFunction::DropOut(OutData, DropOutGradient, mBatchCount * FLinkKernel->ConvKernelCount, DROP_OUT_PARAM);
	// �������Ǳ����1����0�ˣ�һ��ͼƬ���Ѿ�ȫ�������N����
	mOutWidth = mOutHeight = 1;
	mOutChannel = FLinkKernel->ConvKernelCount;
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;

	FLinkKernel->ApplyDropOutGradient(DropOutGradient, mBatchCount * FLinkKernel->ConvKernelCount);
	SAFE_DELETE_ARRAY(DropOutGradient);
	FLinkKernel->ApplyImageReluGradientData(ReluGradient, mBatchCount * FLinkKernel->ConvKernelCount);
	SAFE_DELETE_ARRAY(ReluGradient);
}

float* SdlfCalculatorCPU::SoftMax(SoftMaxKernel* SMK)
{
	// �Ȱ�1 * 1024������ת����1 * 10�ġ���ʵ�����ھ���˷�������һ��1024 * 10�ľ�������ֱ�������򵥵�����ɡ�
	float* OutData = new float[SMK->Column * mBatchCount];
	for (int i = 0; i < mBatchCount; i++)
	{
		for (int j = 0; j < SMK->Column; j++)
		{
			int Pos = i * SMK->Column + j;
			OutData[Pos] = SMK->DotProduct(&(mImageOutData[SMK->Row * i]), j);
		}
		// softmax
		int Pos = i * SMK->Column;
		float Out[10];
		//SdlfFunction::softmax_default(&OutData[Pos], Out, 10);
		SdlfFunction::softmax(&OutData[Pos], Out, 10);
		memcpy(&OutData[Pos], Out, 10 * sizeof(float));
	}
	mOutChannel = SMK->Column;
	// ��ȫ�ǲ��ȵĴ���
	SMK->ApplyLastInput(mImageOutData, mBatchCount * SMK->Row);
	SAFE_DELETE_ARRAY(mImageOutData);
	mImageOutData = OutData;

	return mImageOutData;
}


void SdlfCalculatorCPU::Release()
{
	delete this;
}

