/***********************************************
 * File: SdlfFunction.h
 *
 * Author: LYG
 * Date: ʮһ�� 2020
 *
 * Purpose:
 *
 * 
 **********************************************/
#pragma once
#include <time.h>

struct  ConvBlocks;
struct ConvKernel;
// �ѳ��ú��������ٷ�װһ�Σ���Ҫ������ԭ��
// 1����Щ������Ҫ���ú�����μ��㣻2����ʱ��д���������һЩ���ú�������Ҫ�����ϣ��������Լ���װһ�Σ�Ҫ���ʱ�������ҡ�
class SdlfFunction
{
public:
	static bool RealEqual(float a, float b, float tolerance = 1e-4);

	// �����ֵ����Ҫ�ǳػ����õ���
	static float MaxInPool(float* Data, int Num, int& MaxIndex);
	// ��ƽ��ֵ
	static float AverageInPool(float* Data, int Num);
	// ���
	static float DotProduct(float* Data1, float* Datae2, int Num);
	// x^y
	static double Pow(double x, double y);
	// e^n
	static double Exp(double n);
	// ln(n)��eΪ�����Ķ�������
	static double ln(double n);
	// lg(n)��10Ϊ�����Ķ�������
	static double lg(double n);
	// log(m,n)��mΪ�����Ķ�����������ʱ��Ҫ�õ����֣����罻���أ�����eΪ�����������ҿ���Щ�ط���2Ϊ����
	static double log_m_n(double m, double n);
	// sigmoid���ǳ����õĺ����������Ǽ�������Ƕ������ʱ���滻softmax
	static double sigmoid(double z);
	// softmax��������������ǳ����õĺ���
	// ������float��Ϊ���룬����Ϊʹ��GPU�����ʱ��ȫ������float������ʹ��doubleûʲô���塣
	// InArray��OutArray������һ����Num
	// InArrayDest��Ŀ������InArrayDest��ʵ�ʼ�����
	// ��������һЩ���⴦�������˳���Ϊ0�������
	static void softmax(float* InArrayDest, float* OutArray, int Num);
	// Ĭ�ϵ�softmax�������������⴦��
	static void softmax_default(float* InArrayDest, float* OutArray, int Num);
	// CrossEntropy�������ء�
	static double CrossEntropy(float* InArray, int* InArrayDest, int Num);

	// ��λ�����
	static float UnitRandom();
	//��Χ�����
	static float RangeRandom(float Min, float Max);
	// dropout
	// level����ʧ����ֵ����0-1֮��
	static void DropOut(float& Data, float level);
	static void DropOut(float* Data, float* GradientData, int Num, float level);

	// ���ﲻ�������˵�depth��������������image��depthһ��
	static void BuildConvArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchCountIndex);
	static void BuildConvArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int ConvWidth, int ConvHeight, int BatchCount);
	// �����ʵ�ܼ򵥣�����Ϊ�˸�����˶��룬��������һ���ķ�ʽ
	static void BuildFullConnectedArrayPerBatch(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchIndex);
	static void BuildFullConnectedArray(ConvBlocks* CB, float* ImageData, int ImageWidth, int ImageHeight, int ImageDepth, int BatchCount);

	// �����е��ƣ�һ��ͼƬ����Ҫ��N������������������N��ͼƬ��������һ��ͼƬ����������Ҳ��һ��ͼƬ��
	// Ϊʲô�ƣ���Ϊ�漰��batch���漰���������ˡ���batch�ٶ࣬����������ǲ����
	static void Conv2D(ConvBlocks* CB, int ConvBlocksPerBatch, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData, int BatchCount);
	static void Conv2DPerImagePerConvKernel(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, int ConvKernelIndex, float* ImageOutData, float* ReluOutData);
	static void Conv2DPerImage(ConvBlocks* CB, int ConvBlocksPerBatch, int BatchIndex, ConvKernel* CK, float* ImageOutData, int ImageSize2D, float* ReluOutData);

	// ����ȫ���ӡ�
	static void ConvFullConnected(ConvBlocks* CB, ConvKernel* CK, float* ImageOutData, float* ReluOutData, int BatchCount);
protected:
	SdlfFunction() {};
	~SdlfFunction() {};
};