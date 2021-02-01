#include "vectorutility.cuh"


void vectorutility_randomInit(int* vector_a, int vector_length)
{
    for(int i = 0; i < vector_length; i++)
    {
        vector_a[i] = rand() % 100;
    }
}

void vectorutility_randomInit(float* vector_a, int vector_length)
{
    for(int i = 0; i < vector_length; i++)
    {
        vector_a[i] = rand() % 100;
    }
}

void vectorutility_unitaryInit(int* vector_a, int vector_length)
{
    for(int i = 0; i < vector_length; i++)
    {
        vector_a[i] = 1;
    }
}

__global__ void vectorutility_add(int* vextor_a, int* vector_b, int* vector_c, int vector_length)
{
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x; // current thread for each particular block execution 
  
    if(thread_id < vector_length)
    {
        vector_c[thread_id] = vextor_a[thread_id] + vector_b[thread_id]; // Boundary check
    }
}

__global__ void vectorutility_addUnifiedMemory(int* vector_a, int* vector_b, int* vector_c, int vector_length)
{
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x; // current thread for each particular block execution 

    if(thread_id < vector_length)
    {
        vector_c[thread_id] = vector_a[thread_id] + vector_b[thread_id];
    }
}

int vectorutility_sumReduction_cpu(int* vector, int vector_length) // sum of vector elements computed using the cpu (single thread)
{
    int result = 0;

    for (int i = 0; i <= vector_length - 1; i++) 
    {
        result += vector[i];
    }

    return result;
}

__global__ void vectorutility_sumReduction(int* vector_a_original, int* vector_a_reduct, clock_t* time) // in this function is computed the time of execution of each block
{
    if (threadIdx.x == 0) // first thread of each particular block in execution, initialize the counter
    {
        time[blockIdx.x] = clock(); 
    }

    __shared__ int partial_sum[SHARED_MEMORY_SIZE_FOR_VECTOR];

    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x; // current thread for each particular block execution 

    partial_sum[threadIdx.x] = vector_a_original[thread_id];

    __syncthreads(); // wait that all threads in this particular gpu block are ready (intellisense of Visual Studio mark this as an error... but it works)

    for(int j = 1; j < blockDim.x; j *= 2)
    {
        if(threadIdx.x % (2 * j) == 0) 
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + j];
        }
    }
     
    __syncthreads(); // wait that all threads in this particular gpu block are executed (Attention! there aren't a global synchronizator before NVidia Pascal architecture...)

    if(threadIdx.x == 0) 
    {
        vector_a_reduct[blockIdx.x] = partial_sum[0]; // write accumulated result on zero position of vector reducted
    }

    if (threadIdx.x == 0) // when first thread return in execution, save counter value in vector of object type "clock_t"
    {
        time[blockIdx.x + gridDim.x] = clock();
    }
}

__global__ void vectorutility_sumReduction2(int* vector_a_original, int* vector_a_reduct, clock_t* time) // optimized version : using chache memory
{
    if (threadIdx.x == 0) // first thread of each particular block in execution, initialize the counter
    {
        time[blockIdx.x] = clock();
    }

    __shared__ int partial_sum[SHARED_MEMORY_SIZE_FOR_VECTOR];

    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x; // actual thread for each particular execution 

    partial_sum[threadIdx.x] = vector_a_original[thread_id];

    __syncthreads(); // wait that all threads in this particular gpu block are ready 

    for(int j = 1; j < blockDim.x; j *= 2)
    {
        int i = 2 * j * threadIdx.x;

        if(i < blockDim.x)
        {
            partial_sum[i] += partial_sum[i + j];
        }
    }

    __syncthreads(); // wait that all threads in this particular gpu block are executed (Attention! there aren't a global synchronizator before NVidia Pascal architecture...)

    if(threadIdx.x == 0)
    {
        vector_a_reduct[blockIdx.x] = partial_sum[0]; // write accumulated result on zero position of vector reducted
    }

    if (threadIdx.x == 0) // when first thread return in execution, save counter value in vector of object type "clock_t"
    {
        time[blockIdx.x + gridDim.x] = clock();
    }
}

__global__ void vectorutility_sumReduction3(int* vector_a_original, int* vector_a_reduct, clock_t* time) // optimized version: with this function conflict beetween thread are avoided, so threads are execute in "effective" parallel mode because executed in different block
{
    if (threadIdx.x == 0) // first thread of each particular block in execution, initialize the counter
    {
        time[blockIdx.x] = clock();
    }

    __shared__ int partial_sum[SHARED_MEMORY_SIZE_FOR_VECTOR]; 

    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x; // current thread for each particular block execution 

    partial_sum[threadIdx.x] = vector_a_original[thread_id]; // writing this cpu memory stored vectors in a shared memory stored vector

    __syncthreads(); // wait that all threads in this particular gpu block are ready 

    for (int j = blockDim.x / 2; j > 0; j >>= 1) // j >>= 1 is a left shit of one position and means j = j/2
    {
        if (threadIdx.x < j)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + j];
        }
    }

    __syncthreads(); // wait that all threads in this particular gpu block are executed (Attention! there aren't a global synchronizator before NVidia Pascal architecture...)

    if (threadIdx.x == 0)
    {
        vector_a_reduct[blockIdx.x] = partial_sum[0]; // write accumulated result on zero position of vector reducted
    }

    if (threadIdx.x == 0) // when first thread return in execution, save counter value in vector of object type "clock_t"
    {
        time[blockIdx.x + gridDim.x] = clock();
    }
}

__device__ void warpReduction(volatile int* sharedMemory, int element) // __device__ annotated function is callable by gpu, in other word: a cpu function nasted in a gpu function
{
    sharedMemory[element] += sharedMemory[element + 32];
    sharedMemory[element] += sharedMemory[element + 16];
    sharedMemory[element] += sharedMemory[element + 8];
    sharedMemory[element] += sharedMemory[element + 4];
    sharedMemory[element] += sharedMemory[element + 2];
    sharedMemory[element] += sharedMemory[element + 1]; // without a syncronization: in this fucntion only one thread in cpu (no gpu) are running! 
}

__global__ void vectorutility_sumReduction4(int* vector_a_original, int* vector_a_reduct, clock_t* time) // optimized version: with this function conflict beetween thread are avoided
{
    if (threadIdx.x == 0) // first thread of each particular block in execution, initialize the counter
    {
        time[blockIdx.x] = clock();
    }

    __shared__ int partial_sum[SHARED_MEMORY_SIZE_FOR_VECTOR];

    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x; // current thread for each particular block execution 

    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    partial_sum[threadIdx.x] = vector_a_original[i] + vector_a_original[i + blockDim.x]; // writing these vectors in a shared memory stored vector

    __syncthreads(); // wait that all threads in all gpu blocks are ready 

    for (int j = blockDim.x / 2; j > 32; j >>= 1)
    {
        if (threadIdx.x < j)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + j];
        }

        __syncthreads(); // wait that all threads in this particular gpu block are executed (Attention! there aren't a global synchronizator before NVidia Pascal architecture...)
    }

    if (threadIdx.x < 32) // only 32 threads are working, others are exluded by this condition
    {
        warpReduction(partial_sum, threadIdx.x);
    }

    if (threadIdx.x == 0)
    {
        vector_a_reduct[blockIdx.x] = partial_sum[0]; 
    }

    if (threadIdx.x == 0) // when first thread return in execution, save counter value in vector of object type "clock_t"
    {
        time[blockIdx.x + gridDim.x] = clock();
    }
}

void vectorutility_errorCheck(int* vector_a, int* vector_b, int* vector_c, int vector_length)
{
    for(int i = 0; i < vector_length; i++)
    {
       /* cout << endl << "element " << i << " = " ; // ...for programm dubug
        cout << endl << "GPU result -->  " <<  vector_c[i] << "  ";
        cout << endl << "CPU result -->  " << (vector_a[i] + vector_b[i]) << endl;*/
  
        assert(vector_c[i] == vector_a[i] + vector_b[i]);
    }
}

void vectorutility_errorCheck2(float* vector_a, float* vector_b, float* vector_c, float scale, int n) 
{
    for(int i = 0; i < n; i++) 
    {
        assert(vector_c[i] == scale * vector_a[i] + vector_b[i]);
    }
}
