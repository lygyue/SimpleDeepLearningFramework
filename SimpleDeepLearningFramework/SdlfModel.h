/***********************************************
 * File: SdlfModel.h
 *
 * Author: LYG
 * Date: 十一月 2020
 *
 * Purpose:
 *
 * 
 **********************************************/
#pragma once

#include <vector>
#include "Sdlf.h"
#include "SdlfLayer.h"
#include "SdlfCalculator.h"

class SdlfModelImpl : public SdlfModel
{
public:
	SdlfModelImpl();
	virtual ~SdlfModelImpl();


	// 保存模型到文件。便于操作，可以把模型保存。
	virtual bool SaveModelToFile(const char* FileName) override;
	virtual bool LoadModelFromFile(const char* FileName) override;

	// 保存或者加载训练结果，用于后续使用
	virtual bool SaveTrainParametersToFile(const char* FileName) override;
	virtual bool LoadTrainParametersFromFile(const char* FileName) override;

	// 执行相关
	virtual void AddModelListener(SdlfModelListener* Listener) override;
	virtual void RemoveModelListener(SdlfModelListener* Listener) override;
	// 这里是unsigned char，底层会自动转换成-1，1之间的float
	// Width，Height，Channel之类的，必须跟第一层layer输入保持一致
	virtual void SetImageParam(int ImageWidth, int ImageHeight, int ImageChannel, int Batch) override;
	virtual void SetProcessType(ProcssType PT) override;
	virtual void StartTrainSession(unsigned char* ImageData, unsigned char* Classification) override;

	// 返回测试正确率，只用于测试，不能用于训练
	virtual float GetAccuracyRate() const override;
	// Layer相关
	virtual SdlfLayer* CreateLayer(SdlfLayerType LT) override;
	// 由于Layer是单向链表，知道第一个，能一步一步执行遍历下去
	virtual void SetFirstLayer(SdlfLayer* Layer) override;
	// 由于需要逆向求导，还需要记录最后的layer，往前逆向
	virtual void SetLastLayer(SdlfLayer* Layer) override;

	// 设置动态step标记。这个不知道是不是合理的，感觉有时候会loss降低之后又提高，估计是step太大了的原因？
	// 这里，设置一个，如果loss变小了，动态变小step
	virtual void SetDynamicStep(bool DynamicStep)override;
	virtual void Release() override;
protected:
	static const int MaxBatchCount = 1024;
	float* mTransformImageData;

	void NotifyMessage(const char* Msg);
	void NotifyTrainComplete();
	void UpdateStep(float Loss);
private:
	ProcssType mProcessType;
	SdlfLayerImpl* mFirstLayer;
	SdlfLayerImpl* mLastLayer;
	std::vector<SdlfModelListener*> mListener;

	SdlfCalculator* mCalculator;

	int mImageWidth;
	int mImageHeight;
	int mImageChannel;
	int mBatchCount;

	// 用于计算测试准确率
	int mCorrectCount;
	int mInCorrectCount;
	float mAccuracyRate;
	bool mDynamicStep;
};