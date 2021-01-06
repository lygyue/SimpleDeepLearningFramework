// SdlfSimpleFullConnectedExample.cpp : 定义控制台应用程序的入口点。
//

#include <vector>
#include <time.h>
#include "stdafx.h"
#include "../SimpleDeepLearningFramework/Sdlf.h"
#include "../SdlfTrainExample/MnistFileManager.h"

#pragma comment(lib, "SimpleDeepLearningFramework.lib")

#define			BATCH_COUNT			32

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

	SdlfLayer* Layer3 = g_Model->CreateLayer(FullConnected);
	Layer3->SetActivationFunc(Relu);
	Layer3->SetFullLinkParameter(28, 28, 1, 1024);
	Layer3->InitialiseConvParamToRamdom(-0.01f, 0.01f);
	Layer3->SetGradientParameter(0.01f);

	g_Model->SetFirstLayer(Layer3);
	g_LayerArray.push_back(Layer3);

	SdlfLayer* Layer4 = g_Model->CreateLayer(SoftMax);
	Layer4->InitialiseConvParamToRamdom(-0.01f, 0.01f);
	Layer3->SetNextLayer(Layer4);
	Layer4->SetPreLayer(Layer3);
	Layer4->SetGradientParameter(0.01f);
	g_LayerArray.push_back(Layer4);

	g_Model->SetLastLayer(Layer4);
	g_Model->SetImageParam(28, 28, 1, BATCH_COUNT);

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