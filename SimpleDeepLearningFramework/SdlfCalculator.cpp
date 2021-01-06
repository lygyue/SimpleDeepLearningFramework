/***********************************************
 * File: SdlfCalculator.cpp
 *
 * Author: LYG
 * Date: 十一月 2020
 *
 * Purpose:
 *
 * 
 **********************************************/

#include "SdlfCalculator.h"
#include "SdlfCalculatorCPU.h"
#include <math.h>

SdlfCalculator* SdlfCalculator::CreateCalculator(SdlfCalculatorType SCT)
{
	SdlfCalculator* SC = nullptr;
	if (SCT == SCT_CPU)
	{
		SC = new SdlfCalculatorCPU;
	}
	else if (SCT == SCT_ComputeShader)
	{

	}
	else if (SCT == SCT_Cuda)
	{

	}
	else if (SCT == SCT_Cudnn)
	{

	}
	return SC;
}


float SdlfCalculator::UCharToFloat(unsigned char C)
{
	// 先转到0，1
	float f = float(C) / 255.0f;

	f = f * 2.0f - 1.0f;

	return f;
}

float SdlfCalculator::UCharToFloat_0_1(unsigned char C)
{
	return float(C) / 255.0f;;
}

unsigned char SdlfCalculator::FloatToUChar(float F)
{
	F = (F + 1.0f) * 0.5f;

	unsigned char C = (unsigned char)floorf((F * 255.0f) + 0.5f);

	return C;
}

unsigned char SdlfCalculator::Float_0_1_ToUChar(float F)
{
	unsigned char C = (unsigned char)floorf((F * 255.0f) + 0.5f);

	return C;
}