/***********************************************
 * File: sdlf.h
 *
 * Author: LYG
 * Date: ʮһ�� 2020
 *
 * Purpose:SDLF�� short for simple deep learning framework.
 *
 * 
 **********************************************/

#pragma once

#define				API_EXPORT				extern "C" __declspec(dllexport)
#define				DLL_CALLCONV			__stdcall

enum SdlfActivationFunc
{
	ActiveNone,
	Sigmoid,
	Tanh,
	Relu,

};

enum PoolType
{
	Pool_Max,
	Pool_Average,
};

enum SdlfLossFunc
{
	LogLoss,
	MSELoss,					// �������������һ���ý�����
	SuqareLoss,
	Adaboost,
	Hinge,
	ExponentialLoss,
	CrossEntropyLoss,			// ������
};

enum SdlfLayerType
{
	Convolution,
	FullConnected,
	FeatureMapAverage,			// ֱ�����ֵ������ȫ�������Կ�Ч��
	SoftMax,

};
// �ֿ������׶Ρ�һ����ѵ���׶Σ�һ���ǲ��Խ׶Ρ����Խ׶β���Ҫ�����󵼸��²�����
enum ProcssType
{
	ProcessTrain,
	ProcessTest,
};

// һ��һ�㣬����˫������ķ�ʽ�������һ��GetNextLayer()����һ�㡣Ŀǰ������̫���ӵ����Σ���Щ�����Լ�Ҳû�˽����
// ���󵼸��²������Ƿ���ģ����Խڵ���Ҫ˫��
struct SdlfLayer 
{
	virtual void SetLayerType(SdlfLayerType LT)													= 0;
	virtual SdlfLayerType GetLayerType() const													= 0;

	virtual SdlfLayer* GetPreLayer()															= 0;
	virtual void SetPreLayer(SdlfLayer* Layer)													= 0;
	virtual SdlfLayer* GetNextLayer()															= 0;
	virtual void SetNextLayer(SdlfLayer* Layer)													= 0;

	// Must be convolution layer
	// Ŀǰ��ʱ��������ά�ľ���ˡ�
	// ���Ǿ�����������
	virtual bool SetConvolutionKernel(int ConvKernelWidth, int ConvKernelHeight, int InWidth, int InHeight, int InChannels, int OutChannels)	= 0;
	// ��ʼ������˵Ĳ��������Ĭ��������ġ���������Ҫ�Ļ����ҿ�Ҳ�г�ʼ���ģ�
	// ֻ����w��b
	virtual void SetConvolutionParam(float W, float B)											= 0;
	virtual void InitialiseConvParamToRamdom(float fMin, float fMax)							= 0;
	virtual bool SetActivationFunc(SdlfActivationFunc AF)										= 0;
	// Ŀǰֻ����2*2�ػ�������������
	virtual bool SetPoolParameter(PoolType PT)													= 0;

	// �����󵼲�����Ŀǰ������һ��step��
	virtual void SetGradientParameter(float Step)												= 0;

	// ȫ���Ӳ�������á���˵���ںܶ���㷨�Ѿ�������ȫ���Ӳ㣬���õ�ȫ����ƽ��ֵ��
	// ����Ҳ����һ�£���һ����ֵ�Ĳ���һ��Ч����������û��ʲô��ͬ
	virtual bool SetFullLinkParameter(int Width, int Height, int Depth, int FullLinkDepth)		= 0;

	virtual void Release()																		= 0;
};

struct SdlfModelListener
{
	virtual void OnMessage(const char* Msg)														= 0;
	virtual void OnTrainComplete()																= 0;
};

// ģ����ء�
struct SdlfModel 
{
	// ����ģ�͵��ļ������ڲ��������԰�ģ�ͱ��档
	virtual bool SaveModelToFile(const char* FileName)											= 0;
	virtual bool LoadModelFromFile(const char* FileName)										= 0;

	// ������߼���ѵ����������ں���ʹ��
	virtual bool SaveTrainParametersToFile(const char* FileName)								= 0;
	virtual bool LoadTrainParametersFromFile(const char* FileName)								= 0;

	// ִ�����
	virtual void AddModelListener(SdlfModelListener* Listener)									= 0;
	virtual void RemoveModelListener(SdlfModelListener* Listener)								= 0;
	// ������unsigned char���ײ���Զ�ת����-1��1֮���float
	// Width��Height��Channel֮��ģ��������һ��layer���뱣��һ��
	virtual void SetImageParam(int ImageWidth, int ImageHeight, int ImageChannel, int Batch)	= 0;
	virtual void SetProcessType(ProcssType PT)													= 0;
	// Ĭ�Ͼ���10�����࣬���ò�����
	virtual void StartTrainSession(unsigned char* ImageData, unsigned char* Classification)		= 0;
	// ���ز�����ȷ�ʣ�ֻ���ڲ��ԣ���������ѵ��
	virtual float GetAccuracyRate()	const														= 0;
	// Layer���
	virtual SdlfLayer* CreateLayer(SdlfLayerType LT)											= 0;
	// ����Layer�ǵ�������֪����һ������һ��һ��ִ�б�����ȥ
	virtual void SetFirstLayer(SdlfLayer* Layer)												= 0;
	// ������Ҫ�����󵼣�����Ҫ��¼����layer����ǰ����
	virtual void SetLastLayer(SdlfLayer* Layer)													= 0;

	// ���ö�̬step��ǡ������֪���ǲ��Ǻ���ģ��о���ʱ���loss����֮������ߣ�������step̫���˵�ԭ��
	// �������һ�������loss��С�ˣ���̬��Сstep
	virtual void SetDynamicStep(bool DynamicStep)												= 0;

	virtual void Release()																		= 0;
};


API_EXPORT void DLL_CALLCONV CreateSdlfModel(SdlfModel** Model);