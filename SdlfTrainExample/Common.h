/***********************************************
 * File: Common.h
 *
 * Author: LYG
 * Date: Ê®¶þÔÂ 2020
 *
 * Purpose:
 *
 * 
 **********************************************/

#pragma once

#define SAFE_DELETE(p)       { if(p) { delete (p);     (p)=NULL; } }
#define SAFE_DELETE_ARRAY(p) { if(p) { delete[] (p);   (p)=NULL; } }
#define SAFE_RELEASE(p)      { if(p) { (p)->Release(); (p)=NULL; } }

#define			SINGLETON(T)																\
public:																						\
	static T* GetInstance();																\
	static void Release();																	\
protected:																					\
		static T* Instance;

#define			SINGLETON_IMPL(T)															\
T* T::Instance = nullptr;																	\
T* T::GetInstance()																			\
{																							\
	if (Instance == nullptr)																\
	{																						\
		Instance = new T();																	\
	}																						\
	return Instance;																		\
}																							\
void T::Release()																			\
{																							\
	SAFE_DELETE(Instance);																	\
}																					

