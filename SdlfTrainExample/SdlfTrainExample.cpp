// SdlfTrainExample.cpp : 定义控制台应用程序的入口点。
//

#include <vector>
#include <time.h>
#include "stdafx.h"
#include "Sdlf.h"
#include "MnistFileManager.h"

#pragma comment(lib, "SimpleDeepLearningFramework.lib")

#define			BATCH_COUNT			64

//#define			ONE_CONV_LAYER		1

SdlfModel* g_Model = nullptr;
bool g_TrainCompleted = false;

std::vector<SdlfLayer*> g_LayerArray;

struct ModelListener : public SdlfModelListener
{
	virtual void OnMessage(const char* Msg)
	{
		printf("%s\n", Msg);
	}
	virtual void OnTrainComplete()
	{
		g_TrainCompleted = true;
	}
};

ModelListener Listener;

void ClearAll()
{
	for (size_t i = 0; i < g_LayerArray.size(); i++)
	{
		g_LayerArray[i]->Release();
	}
	g_LayerArray.clear();
	g_Model->Release();
	g_Model = nullptr;

	MnistFileManager::GetInstance()->Release();
}

int main()
{
	CreateSdlfModel(&g_Model);
	if (g_Model == nullptr)
	{
		printf("Create Sdlf model failed.\n");

		return 0;
	}
	if (MnistFileManager::GetInstance()->LoadTrainDataToMemory() == false)
	{
		printf("Load mnist file failed.\n");
		return 0;
	}

	g_Model->SetProcessType(ProcessTrain);
	g_Model->AddModelListener(&Listener);
#ifndef ONE_CONV_LAYER
	SdlfLayer* Layer1 = g_Model->CreateLayer(Convolution);
	g_Model->SetFirstLayer(Layer1);

	Layer1->SetActivationFunc(Relu);
	Layer1->SetConvolutionKernel(5, 5, 28, 28, 1, 32);
	Layer1->InitialiseConvParamToRamdom(-0.01f, 0.01f);
	Layer1->SetGradientParameter(0.2f);

	g_LayerArray.push_back(Layer1);

	SdlfLayer* Layer2 = g_Model->CreateLayer(Convolution);
	Layer1->SetNextLayer(Layer2);
	Layer2->SetPreLayer(Layer1);

	Layer2->SetActivationFunc(Relu);
	Layer2->SetConvolutionKernel(5, 5, 14, 14, 32, 64);
	Layer2->InitialiseConvParamToRamdom(-0.01f, 0.01f);
	Layer2->SetGradientParameter(0.2f);

	g_LayerArray.push_back(Layer2);

	SdlfLayer* Layer3 = g_Model->CreateLayer(FullConnected);
	Layer3->SetActivationFunc(Relu);
	Layer3->SetFullLinkParameter(7, 7, 64, 1024);
	Layer3->InitialiseConvParamToRamdom(-0.01f, 0.01f);
	Layer3->SetGradientParameter(0.05f);

	Layer2->SetNextLayer(Layer3);
	Layer3->SetPreLayer(Layer2);
	g_LayerArray.push_back(Layer3);

	SdlfLayer* Layer4 = g_Model->CreateLayer(SoftMax);
	Layer4->InitialiseConvParamToRamdom(-0.01f, 0.01f);
	Layer3->SetNextLayer(Layer4);
	Layer4->SetPreLayer(Layer3);
	Layer4->SetGradientParameter(0.05f);
	g_LayerArray.push_back(Layer4);

	g_Model->SetLastLayer(Layer4);
	g_Model->SetImageParam(28, 28, 1, BATCH_COUNT);
	g_Model->SetDynamicStep(true);
#else
	SdlfLayer* Layer1 = g_Model->CreateLayer(Convolution);
	g_Model->SetFirstLayer(Layer1);

	Layer1->SetActivationFunc(Relu);
	Layer1->SetConvolutionKernel(5, 5, 28, 28, 1, 32);
	Layer1->InitialiseConvParamToRamdom(-0.01f, 0.01f);
	Layer1->SetGradientParameter(0.01f);

	g_LayerArray.push_back(Layer1);

	SdlfLayer* Layer3 = g_Model->CreateLayer(FullConnected);
	Layer3->SetActivationFunc(Relu);
	Layer3->SetFullLinkParameter(14, 14, 32, 1024);
	Layer3->InitialiseConvParamToRamdom(-0.01f, 0.01f);
	Layer3->SetGradientParameter(0.05f);

	Layer1->SetNextLayer(Layer3);
	Layer3->SetPreLayer(Layer1);
	g_LayerArray.push_back(Layer3);

	SdlfLayer* Layer4 = g_Model->CreateLayer(SoftMax);
	Layer4->InitialiseConvParamToRamdom(-0.01f, 0.01f);
	Layer3->SetNextLayer(Layer4);
	Layer4->SetPreLayer(Layer3);
	Layer4->SetGradientParameter(0.05f);
	g_LayerArray.push_back(Layer4);

	g_Model->SetLastLayer(Layer4);
	g_Model->SetImageParam(28, 28, 1, BATCH_COUNT);
#endif
	srand(time(NULL));
	while (!g_TrainCompleted)
	{
		unsigned char* ImageData, *LabelData;
		MnistFileManager::GetInstance()->GetImageAndLabelDataInRandom(BATCH_COUNT, ImageData, LabelData);

		g_Model->StartTrainSession(ImageData, LabelData);
	}

	ClearAll();
	system("pause");
    return 0;
}