/***********************************************
 * File: SdlfModel.h
 *
 * Author: LYG
 * Date: ʮһ�� 2020
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


	// ����ģ�͵��ļ������ڲ��������԰�ģ�ͱ��档
	virtual bool SaveModelToFile(const char* FileName) override;
	virtual bool LoadModelFromFile(const char* FileName) override;

	// ������߼���ѵ����������ں���ʹ��
	virtual bool SaveTrainParametersToFile(const char* FileName) override;
	virtual bool LoadTrainParametersFromFile(const char* FileName) override;

	// ִ�����
	virtual void AddModelListener(SdlfModelListener* Listener) override;
	virtual void RemoveModelListener(SdlfModelListener* Listener) override;
	// ������unsigned char���ײ���Զ�ת����-1��1֮���float
	// Width��Height��Channel֮��ģ��������һ��layer���뱣��һ��
	virtual void SetImageParam(int ImageWidth, int ImageHeight, int ImageChannel, int Batch) override;
	virtual void SetProcessType(ProcssType PT) override;
	virtual void StartTrainSession(unsigned char* ImageData, unsigned char* Classification) override;

	// ���ز�����ȷ�ʣ�ֻ���ڲ��ԣ���������ѵ��
	virtual float GetAccuracyRate() const override;
	// Layer���
	virtual SdlfLayer* CreateLayer(SdlfLayerType LT) override;
	// ����Layer�ǵ�������֪����һ������һ��һ��ִ�б�����ȥ
	virtual void SetFirstLayer(SdlfLayer* Layer) override;
	// ������Ҫ�����󵼣�����Ҫ��¼����layer����ǰ����
	virtual void SetLastLayer(SdlfLayer* Layer) override;

	// ���ö�̬step��ǡ������֪���ǲ��Ǻ���ģ��о���ʱ���loss����֮������ߣ�������step̫���˵�ԭ��
	// �������һ�������loss��С�ˣ���̬��Сstep
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

	// ���ڼ������׼ȷ��
	int mCorrectCount;
	int mInCorrectCount;
	float mAccuracyRate;
	bool mDynamicStep;
};