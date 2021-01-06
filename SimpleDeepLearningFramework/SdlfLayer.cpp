/***********************************************
 * File: SdlfLayer.cpp
 *
 * Author: LYG
 * Date: ʮһ�� 2020
 *
 * Purpose:
 *
 * 
 **********************************************/

#include "Common.h"
#include "SdlfLayer.h"

SdlfLayerImpl::SdlfLayerImpl()
{
	mLayerType								= Convolution;
	mActivationFunc							= Relu;
	mPoolType								= Pool_Max;
	mInWidth								= 0;
	mInHeight								= 0;
	mInChannel								= 0;
	mPadding								= 0;
	mStep									= STEP;
	mPreLayer								= nullptr;
	mNextLayer								= nullptr;
	mConvKernel								= nullptr;
	mFullLinkKernel							= nullptr;
	mSoftMaxKernel							= nullptr;
}

SdlfLayerImpl::~SdlfLayerImpl()
{
	SAFE_DELETE(mConvKernel);
	SAFE_DELETE(mFullLinkKernel);
	SAFE_DELETE(mSoftMaxKernel);
}

void SdlfLayerImpl::SetLayerType(SdlfLayerType LT)
{
	mLayerType = LT;
}

SdlfLayerType SdlfLayerImpl::GetLayerType() const
{
	return mLayerType;
}

SdlfLayer* SdlfLayerImpl::GetPreLayer()
{
	return mPreLayer;
}

void SdlfLayerImpl::SetPreLayer(SdlfLayer* Layer)
{
	mPreLayer = static_cast<SdlfLayerImpl*>(Layer);
}

SdlfLayer* SdlfLayerImpl::GetNextLayer()
{
	return mNextLayer;
}

void SdlfLayerImpl::SetNextLayer(SdlfLayer* Layer)
{
	mNextLayer = static_cast<SdlfLayerImpl*>(Layer);
}

bool SdlfLayerImpl::SetConvolutionKernel(int ConvKernelWidth, int ConvKernelHeight, int InWidth, int InHeight, int InChannels, int OutChannels)
{
	mInChannel = InChannels;
	mInWidth = InWidth;
	mInHeight = InHeight;

	// ����ٶ�Width��Height��һ���ģ���Ȼ��������Ҳͦ�鷳�������һ����3 * 3�� 5 * 5֮��ġ����磺5 * 5�ľ���ˣ�����һ�߲�2.
	mPadding = (ConvKernelWidth - 1) >> 1;

	mConvKernel = new ConvKernel(ConvKernelWidth, ConvKernelHeight, InChannels, OutChannels);
	return true;
}

void SdlfLayerImpl::SetConvolutionParam(float W, float B)
{
	switch (mLayerType)
	{
	case Convolution:
	{
		if (mConvKernel)
			mConvKernel->InitialiseData(W, B);
		break;
	}
	case FullConnected:
	{
		if (mFullLinkKernel)
			mFullLinkKernel->InitialiseData(W, B);
		break;
	}
	case SoftMax:
	{
		if (!mSoftMaxKernel)
		{
			mSoftMaxKernel = new SoftMaxKernel(1024, 10);
		}
		mSoftMaxKernel->InitialiseData(W, B);
		break;
	}
	}
}

void SdlfLayerImpl::InitialiseConvParamToRamdom(float fMin, float fMax)
{
	switch (mLayerType)
	{
	case Convolution:
	{
		if (mConvKernel)
			mConvKernel->InitialiseDataToRandom(fMin, fMax);
		break;
	}
	case FullConnected:
	{
		if (mFullLinkKernel)
			mFullLinkKernel->InitialiseDataToRandom(fMin, fMax);

		break;
	}
	case SoftMax:
	{
		if (!mSoftMaxKernel)
		{
			mSoftMaxKernel = new SoftMaxKernel(1024, 10);
		}
		mSoftMaxKernel->InitialiseDataToRandom(fMin, fMax);
		break;
	}
	}
}

bool SdlfLayerImpl::SetActivationFunc(SdlfActivationFunc AF)
{
	mActivationFunc = AF;
	return true;
}

bool SdlfLayerImpl::SetPoolParameter(PoolType PT)
{
	mPoolType = PT;
	return true;
}

bool SdlfLayerImpl::SetFullLinkParameter(int Width, int Height, int Depth, int FullLinkDepth)
{
	SAFE_DELETE(mFullLinkKernel);
	mFullLinkKernel = new ConvKernel(Width, Height, Depth, FullLinkDepth);
	mInChannel = Depth;
	mInWidth = Width;
	mInHeight = Height;
	return true;
}

void SdlfLayerImpl::SetGradientParameter(float Step)
{
	mStep = Step;
}

void SdlfLayerImpl::Release()
{
	delete this;
}

float* SdlfLayerImpl::Excute(float* ImageData, int BatchCount, SdlfCalculator* Calculator)
{
	switch (mLayerType)
	{
	case Convolution:
	{
		// ����������ȡ����
		Calculator->Conv2D(ImageData, mInWidth, mInHeight, mInChannel, BatchCount, mConvKernel, mActivationFunc);
		// �ػ���
		Calculator->Max_Pool_2_2(mConvKernel);
		return mNextLayer->Excute(Calculator);
	}
	
	case FullConnected:
	{
		Calculator->CalcFullLink(ImageData, mInWidth, mInHeight, mInChannel, mFullLinkKernel);
		return mNextLayer->Excute(Calculator);
	}
	case FeatureMapAverage:
	{
		break;
	}
	case SoftMax:
	{
		break;
	}
	default:
	{
		break;
	}
	}
	return nullptr;
}

float* SdlfLayerImpl::Excute(SdlfCalculator* Calculator)
{
	switch (mLayerType)
	{
	case Convolution:
	{
		// ����������ȡ����
		Calculator->Conv2D(mConvKernel, mActivationFunc);
		// �ػ���
		Calculator->Max_Pool_2_2(mConvKernel);
		return mNextLayer->Excute(Calculator);
	}
	case FullConnected:
	{
		// ���շ��֣����ȫ���Ӹ�����ļ��㲢�޲�ͬ������ֻ����һ����Ҫ������һ������˾���ͼƬ��ô�󣬲���Ҫ����
		// ȫ���������Ѿ�д����dropout
		Calculator->CalcFullLink(mFullLinkKernel);
		return mNextLayer->Excute(Calculator);
	}
	case FeatureMapAverage:
	{
		break;
	}
	case SoftMax:
	{
		float* Result = Calculator->SoftMax(mSoftMaxKernel);

		return Result;
	}
	default:
	{
		break;
	}
	}
	return nullptr;
}
// �����֪��һ�η���10����д��������GradientData�ĸ�����10 * BatchCount
void SdlfLayerImpl::SoftMaxGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	// �ȼ�����ǰ�������ٸ��²���
	float* Output = new float[BatchCount * mSoftMaxKernel->Row];

	mSoftMaxKernel->CalcGradient(BatchCount, GradientData, Output);
	// ���²���
	mSoftMaxKernel->UpdateParameter(GradientData, BatchCount, mStep);
	// ���ϴ���
	mPreLayer->FullLinkGradient(Output, BatchCount, Calculator);

	SAFE_DELETE_ARRAY(Output);
	return;
}

void SdlfLayerImpl::FullLinkGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	mFullLinkKernel->CalcFullLinkDropOutAndReluGradient(GradientData, BatchCount);

	// ������ǰȫ������
	float* OutData = new float[BatchCount * mFullLinkKernel->ConvKernelWidth * mFullLinkKernel->ConvKernelHeight * mFullLinkKernel->ConvKernelChannel];

	mFullLinkKernel->CalcFullLinkGradient(GradientData, BatchCount, OutData);
	// ����ȫ���Ӳ���
	mFullLinkKernel->UpdateFullLinkParameter(GradientData, BatchCount, mStep);
	// ������ǰ�����
	if(mPreLayer)
		mPreLayer->Conv2DGradient(OutData, BatchCount, Calculator);

	SAFE_DELETE_ARRAY(OutData);
}

void SdlfLayerImpl::Conv2DGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	// GradientData�ĳ������ϴ�MaxPool��w * h * d * BatchCount
	// �����һ�ξ������14 * 14 * 32 * BatchCount���ڶ��ξ����7 * 7 * 64 * BatchCount
	// OutData�ĳ��ȣ���һ����28 * 28 * 32 * BatchCount���ڶ�����14 * 14 * 64 * BatchCount
	float* OutData = new float[mConvKernel->ImageInputWidth * mConvKernel->ImageInputHeight * mConvKernel->ConvKernelCount * BatchCount];
	// ������MaxPool��Relu
	mConvKernel->CalcConv2DMaxPoolAndReluGradient(GradientData, OutData, BatchCount);
	// ��������
	if (mPreLayer)		// ����������жϣ������ǵ�һ���ڵ㡣��һ���ڵ㣬����Ҫ����ǰ��
	{
		mConvKernel->ApplyConvKernelAndRotate180();
		// ������ǰ��
		int Size = mConvKernel->ImageInputWidth * mConvKernel->ImageInputHeight * mConvKernel->ConvKernelChannel * BatchCount;
		float* ConvGradientData = new float[Size];
		memset(ConvGradientData, 0, sizeof(float) * Size);

		mConvKernel->CalcConv2DGradient(OutData, BatchCount, ConvGradientData);
		// �����󵼲���
		mConvKernel->UpdateConv2DParameter(OutData, BatchCount, mStep);
		// ������ǰִ��
		mPreLayer->Conv2DGradient(ConvGradientData, BatchCount, Calculator);

		SAFE_DELETE_ARRAY(ConvGradientData);
	}
	else
	{
		// �����󵼲���
		mConvKernel->UpdateConv2DParameter(OutData, BatchCount, mStep);
	}

	SAFE_DELETE_ARRAY(OutData);
}

void SdlfLayerImpl::UpdateStep(float Multi)
{
	mStep *= Multi;
	if (mNextLayer)
	{
		mNextLayer->UpdateStep(Multi);
	}
}

ConvKernel* SdlfLayerImpl::GetConvKernel()
{
	return mConvKernel;
}

ConvKernel* SdlfLayerImpl::GetFullLinkKernel()
{
	return mFullLinkKernel;
}