/***********************************************
 * File: SdlfLayer.cpp
 *
 * Author: LYG
 * Date: 十一月 2020
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

	// 这里假定Width跟Height是一样的，不然处理起来也挺麻烦。卷积核一般是3 * 3， 5 * 5之类的。例如：5 * 5的卷积核，两边一边补2.
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
		// 计算卷积，提取特征
		Calculator->Conv2D(ImageData, mInWidth, mInHeight, mInChannel, BatchCount, mConvKernel, mActivationFunc);
		// 池化层
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
		// 计算卷积，提取特征
		Calculator->Conv2D(mConvKernel, mActivationFunc);
		// 池化层
		Calculator->Max_Pool_2_2(mConvKernel);
		return mNextLayer->Excute(Calculator);
	}
	case FullConnected:
	{
		// 最终发现，这个全连接跟卷积的计算并无不同，区别只在于一个需要滑动，一个卷积核就是图片那么大，不需要滑动
		// 全连接里面已经写死了dropout
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
// 这个已知是一次分类10个，写死。所以GradientData的个数是10 * BatchCount
void SdlfLayerImpl::SoftMaxGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	// 先计算向前导数，再更新参数
	float* Output = new float[BatchCount * mSoftMaxKernel->Row];

	mSoftMaxKernel->CalcGradient(BatchCount, GradientData, Output);
	// 更新参数
	mSoftMaxKernel->UpdateParameter(GradientData, BatchCount, mStep);
	// 往上传递
	mPreLayer->FullLinkGradient(Output, BatchCount, Calculator);

	SAFE_DELETE_ARRAY(Output);
	return;
}

void SdlfLayerImpl::FullLinkGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	mFullLinkKernel->CalcFullLinkDropOutAndReluGradient(GradientData, BatchCount);

	// 继续向前全连接求导
	float* OutData = new float[BatchCount * mFullLinkKernel->ConvKernelWidth * mFullLinkKernel->ConvKernelHeight * mFullLinkKernel->ConvKernelChannel];

	mFullLinkKernel->CalcFullLinkGradient(GradientData, BatchCount, OutData);
	// 更新全连接参数
	mFullLinkKernel->UpdateFullLinkParameter(GradientData, BatchCount, mStep);
	// 继续向前卷积求导
	if(mPreLayer)
		mPreLayer->Conv2DGradient(OutData, BatchCount, Calculator);

	SAFE_DELETE_ARRAY(OutData);
}

void SdlfLayerImpl::Conv2DGradient(float* GradientData, int BatchCount, SdlfCalculator* Calculator)
{
	// GradientData的长度是上次MaxPool的w * h * d * BatchCount
	// 例如第一次卷积，是14 * 14 * 32 * BatchCount；第二次卷积是7 * 7 * 64 * BatchCount
	// OutData的长度，第一次是28 * 28 * 32 * BatchCount；第二次是14 * 14 * 64 * BatchCount
	float* OutData = new float[mConvKernel->ImageInputWidth * mConvKernel->ImageInputHeight * mConvKernel->ConvKernelCount * BatchCount];
	// 反向求导MaxPool和Relu
	mConvKernel->CalcConv2DMaxPoolAndReluGradient(GradientData, OutData, BatchCount);
	// 反向卷积求导
	if (mPreLayer)		// 必须做这个判断，可能是第一个节点。第一个节点，不需要再往前算
	{
		mConvKernel->ApplyConvKernelAndRotate180();
		// 计算向前求导
		int Size = mConvKernel->ImageInputWidth * mConvKernel->ImageInputHeight * mConvKernel->ConvKernelChannel * BatchCount;
		float* ConvGradientData = new float[Size];
		memset(ConvGradientData, 0, sizeof(float) * Size);

		mConvKernel->CalcConv2DGradient(OutData, BatchCount, ConvGradientData);
		// 更新求导参数
		mConvKernel->UpdateConv2DParameter(OutData, BatchCount, mStep);
		// 继续向前执行
		mPreLayer->Conv2DGradient(ConvGradientData, BatchCount, Calculator);

		SAFE_DELETE_ARRAY(ConvGradientData);
	}
	else
	{
		// 更新求导参数
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