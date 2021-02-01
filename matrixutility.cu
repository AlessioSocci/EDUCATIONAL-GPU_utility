#include "matrixutility.cuh"

void matrixutility_randomInit(int* matrix_a, int matrix_length)
{
    //cout << endl << "matrix initialized with this value: " << endl; // .. for debub purpose

    for(int i = 0; i < matrix_length; i++)
    {
        for(int j = 0; j < matrix_length; j++)
        {
            matrix_a[i * matrix_length + j] = rand() % 100;

            //cout << matrix_a[i * matrix_length + j] << "  "; // .. for debub
        }
    }
}

void matrixutility_1_or_0_init(int* matrix_a, int matrix_length)
{
    //cout << endl << "matrix initialized with this value: " << endl; // .. for debub purpose

    for (int i = 0; i < matrix_length; i++)
    {
        for (int j = 0; j < matrix_length; j++)
        {
            if ((j % 2) == 0)
            {
                matrix_a[i * matrix_length + j] = 0;
            }
            else
            {
                matrix_a[i * matrix_length + j] = 1;    
            }

            //cout << matrix_a[i * matrix_length + j] << "  "; // .. for debub
        }
    }
}

void matrixutility_transpose(int* matrix_a, int* matrix_a_transposed, int matrix_length)
{
    for (int i = 0; i < matrix_length; i++)
    {
        for (int j = 0; j < matrix_length; j++)
        {
            matrix_a_transposed[(j * matrix_length) + i] = matrix_a[(i * matrix_length) + j];
        }
    }
}

__global__ void matrixutility_mul(int* matrix_a, int* matrix_b, int* matrix_c, int matrix_length)
{
    int row = (blockIdx.y * blockDim.y) + threadIdx.y; // change block = change row and column
    int column = (blockIdx.x * blockDim.x) + threadIdx.x; 

    //matrix_c[row * matrix_length + column] = 0;

    if((row < matrix_length) && (column < matrix_length)) // boundary matrix checking
    {
        for (int k = 0; k < matrix_length; k++) // only one cycle in gpu programming !! ...parallelized computation in block change
        {
            matrix_c[row * matrix_length + column] += (matrix_a[row * matrix_length + k]) * (matrix_b[k * matrix_length + column]);
        }
    }
}

__global__ void matrixutility_mul2(int* matrix_a, int* matrix_b, int* matrix_c, int matrix_length, int tile_length) // using tiled matrix and GPU memory chache instead of GPU RAM
{
    int row = blockIdx.y * tile_length + threadIdx.y; // same of version 1 of mul function, but with blockdim = tile_size, the portion of global matrix
    int column = blockIdx.x * tile_length + threadIdx.x;

    __shared__ int tiled_matrix_a[SHARED_MEMORY_SIZE_FOR_MATRIX]; // tiled matrix space allocated in cache memory; shared memory must be a constant equal to the tile matrix dimension (... = tile_length * 2)
    __shared__ int tiled_matrix_b[SHARED_MEMORY_SIZE_FOR_MATRIX];

    for (int i = 0; i < matrix_length / tile_length; i++) // change tile over the global matrix (tile is a portion of entire matrix, so numbero of iteration is global matrix length / size of tile)
    {        
        tiled_matrix_a[(threadIdx.y * tile_length) + threadIdx.x] = matrix_a[(row * matrix_length) + (i * tile_length + threadIdx.x)];
        tiled_matrix_b[(threadIdx.y * tile_length) + threadIdx.x] = matrix_b[(i * tile_length * matrix_length) + (threadIdx.y * matrix_length) + column];

        __syncthreads();

        for (int j = 0; j < tile_length; j++)
        {
            matrix_c[row * matrix_length + column] += tiled_matrix_a[threadIdx.y * blockDim.x + j] * tiled_matrix_b[j * blockDim.x + threadIdx.x];
        }
    
        __syncthreads();
    }
}

void matrixutility_mul3(int* matrix_a, int* matrix_b, int* matrix_c, int matrix_length) // using libraies
{


}

void matrixutility_errorCheck(int* matrix_a, int* matrix_b, int* matrix_c, int matrix_length)
{
    int summa = 0;

    for(int i = 0; i < matrix_length; i++) // cycle for every column of the matrix
    {
        for(int j = 0; j < matrix_length; j++) // cycle for every row of the matrix
        {
            summa = 0;

            for(int k = 0; k < matrix_length; k++) // cycle for every element of the matrix
            {
                summa += matrix_a[i * matrix_length + k] * matrix_b[k * matrix_length + j];
            }

            //cout << endl << endl <<  "row : " << i << endl << "column : " << j; // ...for programm dubug
            //cout << endl << "GPU result -->  " << matrix_c[i * matrix_length + j];
            //cout << endl << "CPU result -->  " << summa << endl;
           
            assert(summa == matrix_c[i * matrix_length + j]);
        }
    }
}