#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <cublas_v2.h>

using namespace std;

const int SHARED_MEMORY_SIZE_FOR_MATRIX = 256; // ...this must be a constant... for this example I chose to divide global matrix in 4 tiled matrix, so shared memory for tiled matrix is 8 x 8 because global matrix is 32 x 32

void matrixutility_randomInit(int* matrix_a, int matrix_dimension);
void matrixutility_1_or_0_init(int* matrix_a, int matrix_length);
void matrixutility_transpose(int* matrix_a, int* matrix_a_transposed, int matrix_length);
__global__ void matrixutility_mul(int* matrix_a, int* matrix_b, int* matrix_c, int matrix_dimension);
__global__ void matrixutility_mul2(int* matrix_a, int* matrix_b, int* matrix_c, int matrix_length, int tile_length);
void matrixutility_errorCheck(int* matrix_a, int* matrix_b, int* matrix_c, int matrix_dimension);