#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <cublas_v2.h>

using namespace std;

const int SHARED_MEMORY_SIZE_FOR_VECTOR = 1 << 12;;

void vectorutility_randomInit(int* vector_a, int vector_length);
void vectorutility_randomInit(float* vector_a, int vector_length); 
void vectorutility_unitaryInit(int* vector_a, int vector_length);
int vectorutility_sumReduction_cpu(int* vector, int vector_length);
__global__ void vectorutility_add(int* vector_a, int* vector_b, int* vector_c, int vector_length);
__global__ void vectorutility_addUnifiedMemory(int* vector_a, int* vector_b, int* vector_c, int vector_length);
__global__ void vectorutility_sumReduction(int* vector_a_original, int* vector_a_reduct, clock_t* time);
__global__ void vectorutility_sumReduction2(int* vector_a_original, int* vector_a_reduct, clock_t* time);
__global__ void vectorutility_sumReduction3(int* vector_a_original, int* vector_a_reduct, clock_t* time);
__global__ void vectorutility_sumReduction4(int* vector_a_original, int* vector_a_reduct, clock_t* time);
void vectorutility_errorCheck(int* vector_a, int* vector_b, int* vector_c, int vector_length);
void vectorutility_errorCheck2(float* vector_a, float* vector_b, float* vector_c, float scale, int n);