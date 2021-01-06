/***********************************************
 * File: sdlf.h
 *
 * Author: LYG
 * Date: 十一月 2020
 *
 * Purpose:SDLF， short for simple deep learning framework.
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
	MSELoss,					// 均方差，分类问题一般用交叉熵
	SuqareLoss,
	Adaboost,
	Hinge,
	ExponentialLoss,
	CrossEntropyLoss,			// 交叉熵
};

enum SdlfLayerType
{
	Convolution,
	FullConnected,
	FeatureMapAverage,			// 直接求均值，不用全连接试试看效果
	SoftMax,

};
// 分开两个阶段。一个是训练阶段，一个是测试阶段。测试阶段不需要反向求导更新参数。
enum ProcssType
{
	ProcessTrain,
	ProcessTest,
};

// 一层一层，采用双向链表的方式。例如第一层GetNextLayer()是下一层。目前不考虑太复杂的情形，那些情形自己也没了解过。
// 而求导更新参数，是反向的，所以节点需要双向。
struct SdlfLayer 
{
	virtual void SetLayerType(SdlfLayerType LT)													= 0;
	virtual SdlfLayerType GetLayerType() const													= 0;

	virtual SdlfLayer* GetPreLayer()															= 0;
	virtual void SetPreLayer(SdlfLayer* Layer)													= 0;
	virtual SdlfLayer* GetNextLayer()															= 0;
	virtual void SetNextLayer(SdlfLayer* Layer)													= 0;

	// Must be convolution layer
	// 目前暂时不考虑三维的卷积核。
	// 这是卷积层参数设置
	virtual bool SetConvolutionKernel(int ConvKernelWidth, int ConvKernelHeight, int InWidth, int InHeight, int InChannels, int OutChannels)	= 0;
	// 初始化卷积核的参数。这个默认是随机的。但是有需要的话，我看也有初始化的？
	// 只设置w跟b
	virtual void SetConvolutionParam(float W, float B)											= 0;
	virtual void InitialiseConvParamToRamdom(float fMin, float fMax)							= 0;
	virtual bool SetActivationFunc(SdlfActivationFunc AF)										= 0;
	// 目前只考虑2*2池化，不考虑其他
	virtual bool SetPoolParameter(PoolType PT)													= 0;

	// 设置求导参数，目前就设置一个step先
	virtual void SetGradientParameter(float Step)												= 0;

	// 全连接层参数设置。据说现在很多的算法已经抛弃了全连接层，而用的全局求平均值。
	// 这里也考虑一下，做一个均值的测试一下效果，看看有没有什么不同
	virtual bool SetFullLinkParameter(int Width, int Height, int Depth, int FullLinkDepth)		= 0;

	virtual void Release()																		= 0;
};

struct SdlfModelListener
{
	virtual void OnMessage(const char* Msg)														= 0;
	virtual void OnTrainComplete()																= 0;
};

// 模型相关。
struct SdlfModel 
{
	// 保存模型到文件。便于操作，可以把模型保存。
	virtual bool SaveModelToFile(const char* FileName)											= 0;
	virtual bool LoadModelFromFile(const char* FileName)										= 0;

	// 保存或者加载训练结果，用于后续使用
	virtual bool SaveTrainParametersToFile(const char* FileName)								= 0;
	virtual bool LoadTrainParametersFromFile(const char* FileName)								= 0;

	// 执行相关
	virtual void AddModelListener(SdlfModelListener* Listener)									= 0;
	virtual void RemoveModelListener(SdlfModelListener* Listener)								= 0;
	// 这里是unsigned char，底层会自动转换成-1，1之间的float
	// Width，Height，Channel之类的，必须跟第一层layer输入保持一致
	virtual void SetImageParam(int ImageWidth, int ImageHeight, int ImageChannel, int Batch)	= 0;
	virtual void SetProcessType(ProcssType PT)													= 0;
	// 默认就是10个分类，不用参数了
	virtual void StartTrainSession(unsigned char* ImageData, unsigned char* Classification)		= 0;
	// 返回测试正确率，只用于测试，不能用于训练
	virtual float GetAccuracyRate()	const														= 0;
	// Layer相关
	virtual SdlfLayer* CreateLayer(SdlfLayerType LT)											= 0;
	// 由于Layer是单向链表，知道第一个，能一步一步执行遍历下去
	virtual void SetFirstLayer(SdlfLayer* Layer)												= 0;
	// 由于需要逆向求导，还需要记录最后的layer，往前逆向
	virtual void SetLastLayer(SdlfLayer* Layer)													= 0;

	// 设置动态step标记。这个不知道是不是合理的，感觉有时候会loss降低之后又提高，估计是step太大了的原因？
	// 这里，设置一个，如果loss变小了，动态变小step
	virtual void SetDynamicStep(bool DynamicStep)												= 0;

	virtual void Release()																		= 0;
};


API_EXPORT void DLL_CALLCONV CreateSdlfModel(SdlfModel** Model);