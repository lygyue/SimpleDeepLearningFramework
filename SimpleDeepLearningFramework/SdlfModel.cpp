/***********************************************
 * File: SdlfModel.cpp
 *
 * Author: LYG
 * Date: ʮһ�� 2020
 *
 * Purpose:
 *
 * 
 **********************************************/

#include "SdlfModel.h"
#include "SdlfFunction.h"

SdlfModelImpl::SdlfModelImpl()
{
	mFirstLayer						= nullptr;
	mLastLayer						= nullptr;
	mImageWidth						= 0;
	mImageHeight					= 0;
	mImageChannel					= 0;
	mBatchCount						= 0;
	mTransformImageData				= nullptr;
	mProcessType					= ProcessTrain;
	mCorrectCount					= 0;
	mInCorrectCount					= 0;
	mAccuracyRate					= 0.0f;
	mDynamicStep					= false;

	mCalculator = SdlfCalculator::CreateCalculator(SCT_CPU);
}


SdlfModelImpl::~SdlfModelImpl()
{
	mCalculator->Release();
	mCalculator = nullptr;

	SAFE_DELETE_ARRAY(mTransformImageData);
}

bool SdlfModelImpl::SaveModelToFile(const char* FileName)
{
	return true;
}

bool SdlfModelImpl::LoadModelFromFile(const char* FileName)
{
	return true;
}


bool SdlfModelImpl::SaveTrainParametersToFile(const char* FileName)
{
	return true;
}

bool SdlfModelImpl::LoadTrainParametersFromFile(const char* FileName)
{
	return true;
}

void SdlfModelImpl::AddModelListener(SdlfModelListener* Listener)
{
	mListener.push_back(Listener);
	return;
}

void SdlfModelImpl::RemoveModelListener(SdlfModelListener* Listener)
{
	for (size_t i = 0; i < mListener.size(); i++)
	{
		if (mListener[i] == Listener)
		{
			mListener.erase(mListener.begin() + i);
			break;
		}
	}
	return;
}

void SdlfModelImpl::SetImageParam(int ImageWidth, int ImageHeight, int ImageChannel, int Batch)
{
	mImageWidth = ImageWidth;
	mImageHeight = ImageHeight;
	mImageChannel = ImageChannel;
	mBatchCount = Batch;

	mTransformImageData = new float[ImageWidth * ImageHeight * ImageChannel * mBatchCount];
}

void SdlfModelImpl::SetProcessType(ProcssType PT)
{
	mProcessType = PT;
}

void SdlfModelImpl::StartTrainSession(unsigned char* ImageData, unsigned char* Classification)
{
	if (mFirstLayer)
	{
		// ��Ҫ�Ȱ�����ת����float����
		int Len = mImageWidth * mImageHeight * mImageChannel * mBatchCount;

		mCalculator->Transform_uchar_to_float(ImageData , mTransformImageData, Len);

		// ��ʼѵ��
		float* Result = mFirstLayer->Excute(mTransformImageData, mBatchCount, mCalculator);

		if (mProcessType == ProcessTrain)
		{
			// ִ����ɣ��󵼣����²���
			// �Ȱѷ�����ת��float�������ź���loss������
			// ��������loss������ƫ�����������������ܼ򵥣����������󵼹��̱Ƚϸ��ӣ�������ϸ�Ƶ���һ��ġ�������Կ����https://zhuanlan.zhihu.com/p/25723112
			float* Classify = new float[10 * mBatchCount];
			float Errors = 0.0f;
			float Loss = 0.0f;
			for (int i = 0; i < mBatchCount * 10; i++)
			{
				Classify[i] = Result[i] - float(Classification[i]);
				if (Classification[i] == 1)
				{
					Loss += -SdlfFunction::ln(Result[i]);
				}
			}
			// ����׼ȷ�ʡ�����ÿ��ѵ������ȷ�ʣ�Ȼ����ͣ�����batch size
			for (int i = 0; i < mBatchCount; i++)
			{
				int MaxIndex = 0;
				SdlfFunction::MaxInPool(Result + i * 10, 10, MaxIndex);
				if (Classification[i * 10 + MaxIndex] == 1)
				{
					Errors += 1.0f;
				}
			}
			mLastLayer->SoftMaxGradient(Classify, mBatchCount, mCalculator);
			SAFE_DELETE_ARRAY(Classify);
			char Msg[256] = { 0 };
			Errors /= float(mBatchCount);
			Loss /= float(mBatchCount);
			if(mDynamicStep)
				UpdateStep(Loss);
			static int TrainTimes = 0;
			sprintf_s(Msg, 256, "Train Count: %d,Train AccuracyRate:%f, Loss:%f\n", TrainTimes++, Errors, Loss);
			NotifyMessage(Msg);
			if (fabs(Errors) >= 0.95f)
			{
				NotifyTrainComplete();
			}
		}
		else if (mProcessType == ProcessTest)
		{
			// ֱ�ӷ������ͽ������ʵ�ʽ�����գ�����׼ȷ�ʡ����⣬���ﲻ����batch��batch count����Ϊ1
			// ȡ���ֵ������������
			int MaxIndex = 0;
			SdlfFunction::MaxInPool(Result, 10, MaxIndex);
			if (Classification[MaxIndex] == 1)
			{
				mCorrectCount++;
			}
			else
			{
				mInCorrectCount++;
			}
			mAccuracyRate = float(mCorrectCount) / float(mCorrectCount + mInCorrectCount);
		}
	}
}

float SdlfModelImpl::GetAccuracyRate() const
{
	return mAccuracyRate;
}

SdlfLayer* SdlfModelImpl::CreateLayer(SdlfLayerType LT)
{
	SdlfLayerImpl* L = new SdlfLayerImpl;
	L->SetLayerType(LT);
	return L;
}

void SdlfModelImpl::SetFirstLayer(SdlfLayer* Layer)
{
	mFirstLayer = static_cast<SdlfLayerImpl*>(Layer);
	return;
}

void SdlfModelImpl::SetLastLayer(SdlfLayer* Layer)
{
	mLastLayer = static_cast<SdlfLayerImpl*>(Layer);
	return;
}

void SdlfModelImpl::SetDynamicStep(bool DynamicStep)
{
	mDynamicStep = DynamicStep;
}

void SdlfModelImpl::Release()
{
	delete this;
}

void SdlfModelImpl::NotifyMessage(const char* Msg)
{
	for (size_t i = 0; i < mListener.size(); i++)
	{
		mListener[i]->OnMessage(Msg);
	}
}

void SdlfModelImpl::NotifyTrainComplete()
{
	for (size_t i = 0; i < mListener.size(); i++)
	{
		mListener[i]->OnTrainComplete();
	}
}

void SdlfModelImpl::UpdateStep(float Loss)
{
	static bool UpdatePhase[3] = { false, false, false };
	if (Loss < 1.5f)
	{
		if (UpdatePhase[0] == false)
		{
			mFirstLayer->UpdateStep(0.1f);
			UpdatePhase[0] = true;
		}
	}
	if (Loss < 1.0f)
	{
		if (UpdatePhase[1] == false)
		{
			mFirstLayer->UpdateStep(0.5f);
			UpdatePhase[1] = true;
		}
	}
	if (Loss < 0.5f)
	{
		if (UpdatePhase[2] == false)
		{
			mFirstLayer->UpdateStep(0.5f);
			UpdatePhase[2] = true;
		}
	}
}

void DLL_CALLCONV CreateSdlfModel(SdlfModel** Model)
{
	SdlfModelImpl* M = new SdlfModelImpl;

	*Model = M;

	return;
}