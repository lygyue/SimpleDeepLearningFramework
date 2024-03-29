/***********************************************
 * File: Common.h
 *
 * Author: LYG
 * Date: 十一月 2020
 *
 * Purpose:
 *
 * 
 **********************************************/
#pragma once

#include <string.h>
#include "SdlfFunction.h"

#define SAFE_DELETE(p)       { if(p) { delete (p);     (p)=nullptr; } }
#define SAFE_DELETE_ARRAY(p) { if(p) { delete[] (p);   (p)=nullptr; } }
#define SAFE_RELEASE(p)      { if(p) { (p)->Release(); (p)=nullptr; } }

#define				DROP_OUT_PARAM				0.3f
#define				STEP						0.1f


#define APPLY_DATA(V)														\
void Apply##V(float* Data, int Len)											\
{																			\
	if(V == nullptr)														\
	{																		\
		V = new float[Len];													\
	}																		\
	memcpy(V, Data, Len * sizeof(float));									\
}

// 其实有非常多更优的方案，例如内存池，矩阵形式等等。但是现在这个方案，让代码可读性更高，便于理解
// 之前尝试用vector容器，那个可读性更高，但是效率极其低下，难以忍受
struct  ConvBlocks
{
	float* ConvArray;
	int ArrayLen;

	ConvBlocks()
	{
		ConvArray = nullptr;
		ArrayLen = 0;
	}

	~ConvBlocks()
	{
		Reset();
	}
	void Reset()
	{
		SAFE_DELETE_ARRAY(ConvArray);
		ArrayLen = 0;
	}
};

struct ConvKernel 
{
	int ConvKernelWidth;
	int ConvKernelHeight;
	int ConvKernelChannel;

	// 这里，为什么有这个？很简单，卷积核，是三维的。例如一个5 * 5 * 32的卷积核。现在要有64个，实际上是5 * 5 * 32 * 64。这样，这个我定义为个数。实际上不知道别人是怎么定义的。但是计算上应该是这样的没有错。
	// 例如，一个5 * 5 * 32的卷积核，实际上是5 * 5 * 1的卷积核，有32个。所以实际是5 * 5 * 1 * 32
	// 也许定义为：InChannel跟OutChannel更加合理一些？
	int ConvKernelCount;

	float* W;
	float* B;

	// 下面的是反向求导用到的
	// 知道宽和高，就知道了relu函数，maxpool函数的宽高，也就知道了很多东西
	int ImageInputWidth;
	int ImageInputHeight;
	ConvBlocks* CB;
	float* WRotate180;				// 反向求导的时候用到，旋转180度
	float* DropOutGradient;
	float* ImageMaxPoolGradientData;
	float* ImageReluGradientData;

	ConvKernel(int Width, int Height, int Depth, int Count)
	{
		ConvKernelWidth = Width;
		ConvKernelHeight = Height;
		ConvKernelChannel = Depth;
		ConvKernelCount = Count;
		W = new float[Width * Height * Depth * Count];
		WRotate180 = new float[Width * Height * Depth * Count];
		B = new float[Count];
		CB = nullptr;
		ImageInputWidth = 0;
		ImageInputHeight = 0;
		DropOutGradient = nullptr;
		ImageMaxPoolGradientData = nullptr;
		ImageReluGradientData = nullptr;
	}
	~ConvKernel()
	{
		SAFE_DELETE_ARRAY(W);
		SAFE_DELETE_ARRAY(WRotate180);
		SAFE_DELETE_ARRAY(B);
		SAFE_DELETE_ARRAY(DropOutGradient);
		SAFE_DELETE_ARRAY(ImageMaxPoolGradientData);
		SAFE_DELETE_ARRAY(ImageReluGradientData);

		SAFE_DELETE(CB);
	}
	APPLY_DATA(DropOutGradient);
	APPLY_DATA(ImageMaxPoolGradientData);
	APPLY_DATA(ImageReluGradientData);

	void ApplyImageInputWidthAndHeight(int Width, int Height)
	{
		ImageInputWidth = Width;
		ImageInputHeight = Height;
	}
	void CalcConv2DMaxPoolAndReluGradient(float* GradientData, float* OutputData)
	{
		for (int i = 0; i < ImageInputHeight / 2; i++)
		{
			for (int j = 0; j < ImageInputWidth / 2; j++)
			{
				int Pos1 = i * 2 * ImageInputWidth + j * 2;
				int Pos2 = i * 2 * ImageInputWidth + j * 2 + 1;
				int Pos3 = (i * 2 + 1) * ImageInputWidth + j * 2;
				int Pos4 = (i * 2 + 1) * ImageInputWidth + j * 2 + 1;

				int Pos = i * ImageInputWidth / 2 + j;
				OutputData[Pos1] = GradientData[Pos] * ImageMaxPoolGradientData[Pos1] * ImageReluGradientData[Pos1];
				OutputData[Pos2] = GradientData[Pos] * ImageMaxPoolGradientData[Pos2] * ImageReluGradientData[Pos2];
				OutputData[Pos3] = GradientData[Pos] * ImageMaxPoolGradientData[Pos3] * ImageReluGradientData[Pos3];
				OutputData[Pos4] = GradientData[Pos] * ImageMaxPoolGradientData[Pos4] * ImageReluGradientData[Pos4];
			}
		}
	}
	// 计算MaxPool和Relu向前求导，两个一起写是因为这样会快点
	// GradientData的大小是ImageInputWidth * ImageInputHeight * ConvKernelCount * BatchCount / 4
	// 除以4是因为输入图片MaxPool之后，宽高其实一边除以2
	// MaxPool反向求导之后，得到的数据会变大四倍。所以输出大小是ImageInputWidth * ImageInputHeight * ConvKernelCount * BatchCount
	void CalcConv2DMaxPoolAndReluGradient(float* GradientData, float* OutputData, int BatchCount)
	{
		int ImageSize = ImageInputHeight * ImageInputWidth;
		for (int i = 0; i < BatchCount * ConvKernelCount; i++)
		{
			CalcConv2DMaxPoolAndReluGradient(&GradientData[ImageSize * i / 4], &OutputData[ImageSize * i]);
		}
	}
	// 赋值卷积核导数，需要把原卷积核旋转180度。卷积核大小等参数必然相等。
	void ApplyConvKernelAndRotate180()
	{
		int HalfConv = ConvKernelWidth / 2;
		int ImageSize = ConvKernelHeight * ConvKernelWidth;
		for (int i = 0; i < ConvKernelCount; i++)
		{
			for (int j = 0; j < ConvKernelChannel; j++)
			{
				for (int m = 0; m < ConvKernelHeight; m++)
				{
					for (int n = 0; n < ConvKernelWidth; n++)
					{
						// 由于旋转是以中心点旋转，所以需要以中心点为坐标原点，建立坐标系。
						// 用矩阵推算一下旋转，其实很简单，假设原本坐标是(x,y)，旋转180度的新坐标是(-x, -y),cos(180) = -1
						int x = n - HalfConv;
						int y = m - HalfConv;
						int _x = -x + HalfConv;
						int _y = -y + HalfConv;
						int Rotate180Pos = ImageSize * ConvKernelChannel * i + ImageSize * j + ConvKernelWidth * _y + _x;
						int OriginalPos = ImageSize * ConvKernelChannel * i + ImageSize * j + ConvKernelWidth * m + n;
						WRotate180[OriginalPos] = W[Rotate180Pos];
					}
				}
			}
		}
	}
#if 0
	// OutData的大小，是ImageInputWidth * ImageInputHeight * ConvKernelChannel
	void CalcConv2DGradientPerImage(float* GradientData, float* OutData, int WIndex)
	{
		float* QuadCache = new float[ConvKernelWidth * ConvKernelHeight];
		int HalfKernel = ConvKernelWidth >> 1;
		int ConvKernelSize = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel;
		float* WRotate_P = WRotate180 + ConvKernelSize * WIndex;
		for (int i = 0; i < ImageInputHeight; i++)
		{
			for (int j = 0; j < ImageInputWidth; j++)
			{
				// 构造QuadCache
				int QuadIndex = 0;
				for (int m = 0; m < ConvKernelHeight; m++)
				{
					for (int n = 0; n < ConvKernelWidth; n++)
					{
						int x = n - HalfKernel;
						int x_ = j + x;
						int y = m - HalfKernel;
						int y_ = i + y;
						if (x_ < 0 || x_ >= ImageInputWidth || y_ < 0 || y_ >= ImageInputHeight)
						{
							QuadCache[QuadIndex++] = 0.0f;
						}
						else
						{
							QuadCache[QuadIndex++] = GradientData[y_ * ImageInputWidth + x_];
						}
					}
				}
				// 遍历卷积核，做点积计算
				for (int m = 0; m < ConvKernelChannel; m++)
				{
					float Sum = 0.0f;
					int KSize = ConvKernelWidth * ConvKernelHeight;
					for (int n = 0; n < KSize; n++)
					{
						Sum += (QuadCache[n] * WRotate_P[m * KSize + n]);
					}
					// 保存到Output
					int Pos = m * ImageInputWidth * ImageInputHeight + i * ImageInputWidth + j;
					OutData[Pos] += Sum;
				}
			}
		}

		SAFE_DELETE_ARRAY(QuadCache);
	}

	// 卷积向前求导
	// 向前求导，需要先把卷积核旋转180度。详情文档里面会有详细介绍。
	// 这里，假设已经旋转完成，并且保存到求导卷积核
	// 卷积的时候，无论卷积核的channel是否为1，必然输出一张depth为1的图片。（这里只针对这个训练，其他情况不知道）
	// 因此，卷积求导的时候，需要先用一张图片，滑动乘以旋转后的卷积核，得到N（N是卷积核的channel）张图片。
	// 举例说明：假设现在有14 * 14 * 32的图片，卷积核是5 * 5 * 32 * 64，卷积之后，会生成14 * 14 * 64的图片。
	// 所以反向求导，是64张14 * 14的图片，去卷积64个5 * 5 * 32的卷积核（这里的卷积核，是旋转过的）。得到64张14 * 14 * 32的图片。最后64张图片再求和，得到一张。
	void CalcConv2DGradient(float* GradientData, int BatchCount, float* OutData)
	{
		int Image2DSize = ImageInputWidth * ImageInputHeight;
		int ImageSize = Image2DSize * ConvKernelChannel;

		int ImageIndex = 0;
		for (int i = 0; i < BatchCount; i++)
		{
			for (int j = 0; j < ConvKernelCount; j++)
			{
				CalcConv2DGradientPerImage(GradientData + Image2DSize * ImageIndex, OutData + ImageSize * i, j);
				ImageIndex++;
			}
		}
	}
#endif
	// 以下部分，其实是上面注释掉的的算法重新写了一遍。为什么这么干？主要是觉得下面这样写，代码易读一些。我开始用的容器，后来发现容器效率太低了，改回了裸指针，其实效率还不如上面的算法。说多了都是泪。
	// OutData的大小，是ImageInputWidth * ImageInputHeight * ConvKernelChannel
	void CalcConv2DGradientPerImage(float* GradientData, float* OutData, int WIndex)
	{
		// 单张输出图片，这里需要卷积一个卷积核，例如图片是14 * 14，卷积核是5 * 5 * 32。那么，这里需要不断去卷积32个5 * 5的卷积核，得到32张14 * 14图片，连接成一张14 * 14 * 32的图片
		// 最后，需要把64张14 * 14 * 32的图片，求和成一张
		int ConvLayerSize = ConvKernelWidth * ConvKernelHeight;
		int ConvKernelSize = ConvLayerSize * ConvKernelChannel;
		float* WRotate_P = WRotate180 + ConvKernelSize * WIndex;
		ConvBlocks* CB = new ConvBlocks;
		SdlfFunction::BuildConvArray(CB, GradientData, ImageInputWidth, ImageInputHeight, 1, ConvKernelWidth, ConvKernelHeight, 1);
		// 卷积核个数
		int CKCount = ImageInputWidth * ImageInputHeight;
		for (int i = 0; i < ConvKernelChannel; i++)
		{
			int StartIndex = i * ConvKernelWidth * ConvKernelHeight;
			float* WRotate_P_I = WRotate_P + i * ConvLayerSize;

			for (int j = 0; j < CKCount; j++)
			{
				float Sum = SdlfFunction::DotProduct(WRotate_P_I, &CB->ConvArray[j * CB->ArrayLen], ConvLayerSize);
				OutData[i * CKCount + j] += Sum;
 			}
		}

		SAFE_DELETE(CB);
	}

	void CalcConv2DGradientPerBatch(float* GradientData, int BatchIndex, float* OutData)
	{
		// 向前卷积的时候，一个Batch，输出ConvKernelCount那个多个图片，这里反过来，需要遍历所有图片，反向计算出一个batch的数据，然后求和
		int Image2DSize = ImageInputWidth * ImageInputHeight;
		for (int i = 0; i < ConvKernelCount; i++)
		{
			CalcConv2DGradientPerImage(GradientData + Image2DSize * i, OutData, i);
		}
	}
	// 卷积向前求导
	// 向前求导，需要先把卷积核旋转180度。详情文档里面会有详细介绍。
	// 这里，假设已经旋转完成，并且保存到求导卷积核
	// 卷积的时候，无论卷积核的channel是否为1，必然输出一张depth为1的图片。（这里只针对这个训练，其他情况不知道）
	// 因此，卷积求导的时候，需要先用一张图片，滑动乘以旋转后的卷积核，得到N（N是卷积核的channel）张图片。
	// 举例说明：假设现在有14 * 14 * 32的图片，卷积核是5 * 5 * 32 * 64，卷积之后，会生成14 * 14 * 64的图片。
	// 所以反向求导，是64张14 * 14的图片，去卷积64个5 * 5 * 32的卷积核（这里的卷积核，是旋转过的）。得到64张14 * 14 * 32的图片。最后64张图片再求和，得到一张。
	void CalcConv2DGradient(float* GradientData, int BatchCount, float* OutData)
	{
		int Image2DSize = ImageInputWidth * ImageInputHeight;
		int ImageSize = Image2DSize * ConvKernelChannel;
		int ImageOutBatchSize = Image2DSize * ConvKernelCount;
		int ImageInBatchSize = ImageSize;

		for (int i = 0; i < BatchCount; i++)
		{
			CalcConv2DGradientPerBatch(GradientData + ImageOutBatchSize * i, i, OutData + ImageInBatchSize * i);
		}
	}
	// 卷积的时候，每一张图片，都是一张输入图片（例如14 * 14 * 32），卷积一个卷积核（例如5 * 5 * 32 * 64）得到的。
	// 一张输入图片，这里按道理得到64张输出图片。这里每一张图片，按道理需要去更新对应位置的卷积核：例如第5张图片，去更新第五个卷积核的参数
	// 由于卷积次数非常多，例如一张图片，会卷积Width * Height那么多次，所以一个参数（例如B，需要被更新这么多次）
	// 此外，还需要考虑BatchSize
	void UpdateConv2DParameterPerImage(int BatchCount, int BatchIndex, int ConvIndex, float* GradientData, float step)
	{
		int BlockCount = ImageInputWidth * ImageInputHeight;
		int StartIndex = BlockCount * BatchIndex;
		int ConvBlockSize = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel;
		int ImageSize = BlockCount;
		for (int i = 0; i < ImageSize; i++)
		{
			B[ConvIndex] -= (step * GradientData[i] / float(BatchCount));
			for (int j = 0; j < ConvBlockSize; j++)
			{
				// 每个像素需要分别乘以一个卷积核那么大小的一个小块
				W[ConvIndex * ConvBlockSize + j] -= (step * GradientData[i] * CB->ConvArray[(StartIndex + i) * CB->ArrayLen + j] / float(BatchCount));
			}			
		}
	}
	// 更新卷积参数
	// GradientData大小：ImageInputWidth * ImageInputHeight * ConvKernelCount * BatchCount
	// 这个，可以看成是ConvKernelCount * BatchCount那么多张图片，每张图片的每个像素，都要更新一次参数。
	// 每张图片的大小，其实是ImageInputWidth * ImageInputHeight * ConvKernelChannel
	// 更新的方式，其实就是求偏导，点乘的方式，点乘输入的图片对应位置的像素。
	// 例如，卷积核是14 * 14 * 32的，有64个，每张图片每个像素，需要更新14 * 14 * 32个参数。
	// 由于，输出特征的像素非常多，每个像素，都需要对输入图片做一个滑块点乘，
	// 因此，最优的方案，提前先把输入图片，分割成一个一个卷积核大小的小块，遍历每一个输出图片像素的时候，像素与所有的小块相乘，再去更新卷积核参数
	// 超出图片边缘的像素补0。
	void UpdateConv2DParameter(float* GradientData, int BatchCount, float step)
	{
		// 第一步，先把输入图片，分成一个一个小块。每个小块的大小，是ConvKernelWidth * ConvKernelHeight * ConvKernelChannel
		// 一张图片能分割的大小，是ImageInputWidth * ImageInputHeight那么多个，这里需要忽略channel，因为本身也是用来做点乘的。
		// 所以，这里分成的小块数，是ImageInputWidth * ImageInputHeight * BatchCount。
		// 我算了一下，使用这种快速方式，一次求导的内存使用，如果是64batch size，需要40M。这是一个相当恐怖的数字了，因为这里是很小的图，只有14 * 14.大图的话，这种方式根本不科学。

		int Image2DSize = ImageInputWidth * ImageInputHeight;
		int ImageBlockSize = Image2DSize * ConvKernelCount;

		// 一张一张图片遍历，去更新W参数，总共有ImageCount那么多张图片
		for (int i = 0; i < BatchCount; i++)
		{
			for (int j = 0; j < ConvKernelCount; j++)
			{
				UpdateConv2DParameterPerImage(BatchCount, i, j, &GradientData[i * ImageBlockSize + j * Image2DSize], step);
			}
		}
	}
	// 放一起算速度快点，反正就一个简单的乘法
	void CalcFullLinkDropOutAndReluGradient(float* GradientData, int BatchCount)
	{
		int Len = BatchCount * ConvKernelCount;
		for (int i = 0; i < Len; i++)
		{
			GradientData[i] = GradientData[i] * ImageReluGradientData[i] * DropOutGradient[i];
		}
		return;
	}

	// 全连接求导
	// 需要把1 * 1024的每个数，乘以一个卷积核，得到Width * Height * Depth的图。一次求导，是1024张图相加。另外，还要考虑到batch count
	// ConvKernelCount == 1024。GradientData的长度是1024 * BatchCount
	// 向前求导，需要考虑到权重。这里的权重，是1024个w求和，所以要除以这个求和
	// OutData大小是Width * Height * Depth * BatchCount.
	// 全连接，不用滑动，卷积核大小跟图片的大小是一样的
	void CalcFullLinkGradient(float* GradientData, int BatchCount, float* OutData)
	{
		// 这里，是7 * 7 * 64
		int ImageSize = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel;
		memset(OutData, 0, sizeof(float) * ImageSize * BatchCount);
		for (int i = 0; i < BatchCount; i++)
		{
			for (int j = 0; j < ConvKernelCount; j++)
			{
				int Pos = i * ConvKernelCount + j;
				for (int k = 0; k < ImageSize; k++)
				{
					OutData[ImageSize * i + k] += (GradientData[Pos] * W[j * ImageSize + k]);
				}
			}
		}
	}

	// 更新全连接参数
	// GradientData的长度是1024 * BatchCount
	// ImageInputData：全连接上次输入计算的数据，实际数据长度是ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * BatchCount
	// 全连接的实质，是一张w * h * d的图片，点乘一个w * h * d的卷积核，变成一个数。全连接现在有1024个卷积核，所以变成了1024个数。
	// 参数更新的实质，是w * x + b求对w求偏导，其实就是x。所以导数其实是GradientData * x。
	// 这里的x，是LastInput。也就是上次池化之后的输出。
	// 更新参数，需要除以batch size
	void UpdateFullLinkParameter(float* GradientData, int BatchCount, float step)
	{
		int ImageSize = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel;
		for (int i = 0; i < BatchCount; i++)
		{
			for (int j = 0; j < ConvKernelCount; j++)
			{
				// 更新B
				B[j] -= step * GradientData[i * ConvKernelCount + j] / float(BatchCount);
				// 更新W
				for (int k = 0; k < ImageSize; k++)
				{
					W[j * ImageSize + k] -= step * GradientData[i * ConvKernelCount + j] * CB->ConvArray[i * CB->ArrayLen + k] / float(BatchCount);
				}
			}

		}
	}

	void InitialiseData(float w, float b)
	{
		int Len = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * ConvKernelCount;
		for (int i = 0; i < Len; i++)
		{
			W[i] = w;
		}
		for (int i = 0; i < ConvKernelCount; i++)
		{
			B[i] = b;
		}
	}

	void InitialiseDataToRandom(float fMin, float fMax)
	{
		int Len = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * ConvKernelCount;
		for (int i = 0; i < Len; i++)
		{
			W[i] = SdlfFunction::RangeRandom(fMin, fMax);
		}
		for (int i = 0; i < ConvKernelCount; i++)
		{
			B[i] = SdlfFunction::RangeRandom(fMin, fMax);
		}
	}

	// 这里，可以是全连接，也可以是卷积，最终计算方式并无不同
	// 这个全连接，其实也就是个点积，例如一张图片，N层之后，输出为7 * 7 * 64。要做一个全连接，那么可以用一个7 * 7 * 64 * 1024，理解成1024个卷积核，每个卷积核跟之前的输出点乘，得到一个数
	// 这样，一张图片，就变成了1024个数。
	float Conv2D(float* Data, int CountIndex)
	{
		float Sum = 0;
		int Pos = ConvKernelWidth * ConvKernelHeight * ConvKernelChannel * CountIndex;
		for (int i = 0; i < ConvKernelWidth * ConvKernelHeight * ConvKernelChannel; i++)
		{
			Sum += W[Pos + i] * Data[i];
		}
		Sum += B[CountIndex];

		return Sum;
	}
};

// 全连接之后，softmax之前，需要把1 * 1024的数据转换成1 * 10的数据，再softmax
// 具体做法就是用个简单的矩阵乘法，乘以一个1024 * 10的矩阵
// 所以这里Row是1024，Column是10
// 这里不太知道应该怎样命名，就先这样吧
struct SoftMaxKernel 
{
	int Row;
	int Column;

	float* W;
	float* B;
	// 反向求导权重计算用到
	float* WSum;
	// 写这里完全是个渣渣代码。但是确实由于前其设计的时候不熟悉，仓促之下写的。实在是想不到更好的方案了。等于前期设计稀烂，一边写一边弥补的感觉：（
	// 这里的大小应该是1024 * BatchCount
	float* LastInput;

	SoftMaxKernel(int R, int C)
	{
		Row = R;
		Column = C;

		W = new float[R * C];
		B = new float[C];
		WSum = new float[R];
		LastInput = nullptr;
	}
	~SoftMaxKernel()
	{
		SAFE_DELETE_ARRAY(W);
		SAFE_DELETE_ARRAY(B);
		SAFE_DELETE_ARRAY(LastInput);
		SAFE_DELETE_ARRAY(WSum);
	}
	APPLY_DATA(LastInput)

	void InitialiseData(float w, float b)
	{
		int Len = Row * Column;
		for (int i = 0; i < Len; i++)
		{
			W[i] = w;
		}
		for (int i = 0; i < Column; i++)
		{
			B[i] = b;
		}
	}

	void InitialiseDataToRandom(float fMin, float fMax)
	{
		int Len = Row * Column;
		for (int i = 0; i < Len; i++)
		{
			W[i] = SdlfFunction::RangeRandom(fMin, fMax);
		}
		for (int i = 0; i < Column; i++)
		{
			B[i] = SdlfFunction::RangeRandom(fMin, fMax);
		}
	}

	float DotProduct(float* Data, int ColIndex)
	{
		float Sum = 0.0f;

		for (int i = 0; i < Row; i++)
		{
			Sum += Data[i] * W[ColIndex * Row + i];
		}
		Sum += B[ColIndex];
		return Sum;
	}
	// 更新参数。根据反向链式求导更新
	// 实际训练的时候，计算过程是sigma(OutputData[i] * W[i]) + B
	// 所以，求导更新参数的时候，其实每个参数W -= step * (GradientData[i] * OutputData[i]);
	// 所以，这里OutputData是之前还没softmax之前的1 * 1024的数据。GradientData是loss求导之后的1 * 10的数据
	// 更新参数，需要除以批次
	void UpdateParameter(float* GradientData, float* OutputData, float step, int BatchCount)
	{
		// 先更新B
		for (int i = 0; i < Column; i++)
		{
			B[i] -= step * GradientData[i] / float(BatchCount);
			// 更新W
			int Pos = i * Row;
			for (int j = 0; j < Row; j++)
			{
				W[Pos + j] -= (OutputData[j] * GradientData[i] * step / float(BatchCount));
			}
		}
	}
	void UpdateParameter(float* GradientData, int BatchCount, float step)
	{
		for (int i = 0; i < BatchCount; i++)
		{
			UpdateParameter(&GradientData[i * Column], &LastInput[i * Row], step, BatchCount);
		}
	}
	// 链式求导，向前推进
	// 链式求导的时候，应该是1 * 10的数据，每一个数，都乘以1024 * 1的数据，得到10个1024 * 1的数组。然后10个1024 * 1的数组相加，得到最终的1 * 1024(1024 * 1)的数组
	// o = w1 * x1 + w2 * x2 + w3 * x3……。对w求导，那就是x，对x求导，那就是w。所以，这里反向求导，就是乘以w
	// GradientData是1 * 10的反向求导数据；OutputData是需要计算得到的1 * 1024的数据
	// 这里，要考虑到权重分配。也就是说，求和的时候，需要除以w1 + w2 + w3 + …… + w10
	void CalcGradient(float* GradientData, float* OutputData)
	{
		memset(OutputData, 0, Row * sizeof(float));
		for (int i = 0; i < Column; i++)
		{
			for (int j = 0; j < Row; j++)
			{
				OutputData[j] += GradientData[i] * W[i * Row + j];
			}
		}
		/*
		for (int j = 0; j < Row; j++)
		{
			OutputData[j] /= WSum[j];
		}
		*/
	}

	void CalcGradient(int BatchCount, float* GradientData, float* OutputData)
	{
		// 开始计算之前，先把参数求和，后面权重计算要用到
		memset(WSum, 0, sizeof(float) * Row);
		for (int i = 0; i < Row; i++)
		{
			for (int j = 0; j < Column; j++)
			{
				WSum[i] += W[j * Row + i];
			}
		}
		for (int i = 0; i < BatchCount; i++)
		{
			CalcGradient(&GradientData[i * Column],&OutputData[i * Row]);
		}
	}
};