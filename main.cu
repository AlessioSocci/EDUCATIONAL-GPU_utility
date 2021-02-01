
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <windows.h>
#include <numeric>
#include <ctime>

#include <cublas.h>

using namespace std;

#include "vectorutility.cuh"
#include "matrixutility.cuh"


int main()
{
    // --- ADD VECTOR ---

    int vector_length = 1 << 16;
    
    size_t memory_space_4vector = sizeof(int) * vector_length; // 65536 elements for each vector

    int threads_quantity = 256;  // set number of threads to execute in a single block
    int blocks_quantity = ((vector_length + threads_quantity - 1) / threads_quantity); // the number of total block selected to equally distribute threads

    int* cpu_ptr_a;
    int* cpu_ptr_b;
    int* cpu_ptr_c;

    int* gpu_ptr_a;
    int* gpu_ptr_b;
    int* gpu_ptr_c;

    cpu_ptr_a = (int*)malloc(memory_space_4vector); // to allocate space in cpu memory (heap)
    cpu_ptr_b = (int*)malloc(memory_space_4vector);
    cpu_ptr_c = (int*)malloc(memory_space_4vector);

    cudaMalloc(&gpu_ptr_a, memory_space_4vector); // to allocate space in gpu memory 
    cudaMalloc(&gpu_ptr_b, memory_space_4vector); 
    cudaMalloc(&gpu_ptr_c, memory_space_4vector);

    vectorutility_randomInit(cpu_ptr_a, vector_length); // assigned random value for each elements of vector
    vectorutility_randomInit(cpu_ptr_b, vector_length);

    cudaMemcpy(gpu_ptr_a, cpu_ptr_a, memory_space_4vector, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored vector in gpu allocated memory
    cudaMemcpy(gpu_ptr_b, cpu_ptr_b, memory_space_4vector, cudaMemcpyHostToDevice);

    vectorutility_add <<<blocks_quantity, threads_quantity>>> (gpu_ptr_a, gpu_ptr_b, gpu_ptr_c, vector_length); // call vectoradd function with cpu vector as parameters, cuda parameters (in <<<...>>>) can be int or dim3 type

    cudaMemcpy(cpu_ptr_c, gpu_ptr_c, memory_space_4vector, cudaMemcpyDeviceToHost); // copy the resulting gpu stored vector, in cpu stored vector

    vectorutility_errorCheck(cpu_ptr_a, cpu_ptr_b, cpu_ptr_c, vector_length); // ...to verify the result by cpu procesing

    free(cpu_ptr_a); // free preallocated cpu memory space (heap)
    free(cpu_ptr_b);
    free(cpu_ptr_c);

    cudaFree(gpu_ptr_a); // free preallocated gpu memory space
    cudaFree(gpu_ptr_b);
    cudaFree(gpu_ptr_c);

    cout << endl;
    printf(" ---> ADD VECTOR - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    // --- ADD VECTOR USING UNIFIED MEMORY ---
    // no need to manage memory

    int used_device = cudaGetDevice(&used_device);

    int* a;
    int* b;
    int* c;

    cudaMallocManaged(&a, memory_space_4vector); // allocate space for vector definition in "unified memory": cpu and gpu memory
    cudaMallocManaged(&b, memory_space_4vector);
    cudaMallocManaged(&c, memory_space_4vector);

    vectorutility_randomInit(a, vector_length); 
    vectorutility_randomInit(b, vector_length);

    cudaMemPrefetchAsync(a, memory_space_4vector, used_device);
    cudaMemPrefetchAsync(b, memory_space_4vector, used_device);

    vectorutility_addUnifiedMemory <<<blocks_quantity, threads_quantity>>> (a, b, c, vector_length); // call vectoradd function with cpu vector as parameters

    cudaDeviceSynchronize(); // wait the end of all previous operation

    cudaMemPrefetchAsync(c, memory_space_4vector, cudaCpuDeviceId);

    vectorutility_errorCheck(a, b, c, vector_length); // ...to verify the result by cpu procesing

    cout << endl;
    printf(" ---> ADD VECTOR IN UNIFIED MEMORY - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    // --- ADD VECTOR USING LIBRARIES ---

    //float* cpu_ptr_1;
    //float* cpu_ptr_2;
    //float* cpu_ptr_3;

    //cpu_ptr_1 = (float*)malloc(MEMORY_SPACE_4VECTOR); // to allocate space in cpu memory (heap)
    //cpu_ptr_2 = (float*)malloc(MEMORY_SPACE_4VECTOR); 
    //cpu_ptr_3 = (float*)malloc(MEMORY_SPACE_4VECTOR);

    //float* gpu_ptr_1; // declare a float pointer type variable
    //float* gpu_ptr_2; 

    //cudaMalloc(&gpu_ptr_1, MEMORY_SPACE_4VECTOR); // to allocate space in gpu memory 
    //cudaMalloc(&gpu_ptr_1, MEMORY_SPACE_4VECTOR);

    //vectorutility_randomInit(cpu_ptr_1, VECTOR_LENGTH); // random initalizing of vector allocated in CPU
    //vectorutility_randomInit(cpu_ptr_2, VECTOR_LENGTH);

    //cublasHandle_t handle; // declare a new cublasHandle object type 
    //cublasCreate_v2(&handle); 

    //cublasSetVector(VECTOR_LENGTH, sizeof(float), cpu_ptr_1, 1, gpu_ptr_1, 1); // copy vector from CPU RAM to GPU RAM
    //cublasSetVector(VECTOR_LENGTH, sizeof(float), cpu_ptr_2, 1, gpu_ptr_2, 1);

    //const float scale = 2.0f;

    //cublasSaxpy_v2(handle, VECTOR_LENGTH, &scale, gpu_ptr_1, 1, gpu_ptr_2, 1);

    //cublasGetVector(VECTOR_LENGTH, sizeof(float), gpu_ptr_2, 1, cpu_ptr_3, 1);

    //vectorutility_errorCheck2(cpu_ptr_1, cpu_ptr_2, cpu_ptr_3, scale, VECTOR_LENGTH);

    //cublasDestroy(handle);

    //cudaFree(cpu_ptr_1); // free preallocated cpu memory space (heap)
    //cudaFree(cpu_ptr_2);
    //cudaFree(cpu_ptr_3);

    //cudaFree(gpu_ptr_1); // free preallocated gpu memory space
    //cudaFree(gpu_ptr_2);
  
    //cout << endl;
    //printf(" ---> ADD VECTOR USING LIBRARY - OK ! <--- "); // print success message in console if there aren't errors 
    //cout << endl;

    //Sleep(2000);
    
    
    // SUM REDUCTION IN CPU

    vector_length = 1 << 16; // 65536 in decimal

    int* vector_a;

    int result = 0;

    vector_a = (int*)malloc(memory_space_4vector); // allocate space in cpu memory (heap)
    
    vectorutility_unitaryInit(vector_a, vector_length);

    clock_t time_cpu; // timer object for cpu clock computation

    time_cpu = clock(); // start timer for cpu clock computation

    result = vectorutility_sumReduction_cpu(vector_a, vector_length); // compute in cpu the result of adding operation between vector elements

    time_cpu = clock() - time_cpu;
    
    cout << endl << "total clock cycle : " << time_cpu << endl;
    cout << "result on cpu computation : " << result;

    free(vector_a);

    cout << endl;
    printf(" ---> VECTOR SUM IN CPU - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    // SUM REDUCTION

    vector_length = 1 << 12;
    memory_space_4vector = sizeof(int) * vector_length; // 4096 elements for each vector

    threads_quantity = 128;  // set number of threads to execute
    blocks_quantity = (int)ceil(vector_length / threads_quantity); // 32 blocks in this example

    vector<int> cpu_ptr_a_original(vector_length); // with use of container vector, no need to allocate space in memory 
    vector<int> cpu_ptr_a_reduct(vector_length);

    clock_t* time = new clock_t[blocks_quantity * 2]; // instanciate a pointer to a vector of type "clock_t" object
    clock_t* d_time;

    int* gpu_ptr_a_original;
    int* gpu_ptr_a_reduct;

    cudaMalloc(&gpu_ptr_a_original, memory_space_4vector); // to allocate space in gpu memory 
    cudaMalloc(&gpu_ptr_a_reduct, memory_space_4vector);
  
    generate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), []() { return 1; }); // assigned value 1 for each elements of vector (containter) using lambda expression (with void return) 

    cudaMemcpy(gpu_ptr_a_original, cpu_ptr_a_original.data(), memory_space_4vector, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored vector in gpu allocated memory

    cudaMalloc(&d_time, sizeof(clock_t) * blocks_quantity * 2);  // allocate space in gpu ememory for time computation

    vectorutility_sumReduction <<<blocks_quantity, threads_quantity>>> (gpu_ptr_a_original, gpu_ptr_a_reduct, d_time); // execute this function 
   
    cudaMemcpy(time, d_time, sizeof(clock_t)* blocks_quantity * 2, cudaMemcpyDeviceToHost);
    
    vectorutility_sumReduction <<<1, threads_quantity>>> (gpu_ptr_a_reduct, gpu_ptr_a_reduct, d_time); 
   
    cout << endl;

    int elapsed_time = 0;

    for (int i = 0; i < blocks_quantity; i++)
    {    
        // cout << i << "  " << (time[i + blocks_quantity] - time[i]) << endl; // .. for debug
        elapsed_time += time[i + blocks_quantity] - time[i];
    }

    cout << endl << "total clock cycle : " << elapsed_time << endl;

    cudaFree(d_time); // free preallocated gpu memory space for timing 

    cudaMemcpy(cpu_ptr_a_reduct.data(), gpu_ptr_a_reduct, memory_space_4vector, cudaMemcpyDeviceToHost);  // copy the resulting gpu stored vector, in cpu stored vector

    cout << "result on gpu computation : " << cpu_ptr_a_reduct[0] << endl; // espected 65536 because all elements are setted at value one, this result is computed in gpu
    cout << "result on cpu computation : " << accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0); // print on console the sum of vector elements, computed in cpu

    assert(cpu_ptr_a_reduct[0] == accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0)); // compare and check the rusults 
   
    cudaFree(gpu_ptr_a_original); // free preallocated gpu memory space
    cudaFree(gpu_ptr_a_reduct);

    cout << endl;
    printf(" ---> VECTOR SUM - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);

    // SUM REDUCTION ACCELERATED

    threads_quantity = 256;  // set number of threads to execute
    blocks_quantity = (int)ceil(vector_length / threads_quantity); // 64 blocks in this example

    cudaMalloc(&gpu_ptr_a_original, memory_space_4vector); // to allocate space in gpu memory 
    cudaMalloc(&gpu_ptr_a_reduct, memory_space_4vector);

    generate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), []() { return 1; }); // assigned value 1 for each elements of vector (containter) using lambda expression (with void return) 

    cudaMemcpy(gpu_ptr_a_original, cpu_ptr_a_original.data(), memory_space_4vector, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored vector in gpu allocated memory

    cudaMalloc(&d_time, sizeof(clock_t) * blocks_quantity * 2);  // allocate space in gpu ememory for time computation

    vectorutility_sumReduction << <blocks_quantity, threads_quantity >> > (gpu_ptr_a_original, gpu_ptr_a_reduct, d_time); // execute this function 

    cudaMemcpy(time, d_time, sizeof(clock_t) * blocks_quantity * 2, cudaMemcpyDeviceToHost);

    vectorutility_sumReduction << <1, threads_quantity >> > (gpu_ptr_a_reduct, gpu_ptr_a_reduct, d_time);

    cout << endl;

    elapsed_time = 0;

    for (int i = 0; i < blocks_quantity; i++)
    {
        // cout << i << "  " << (time[i + blocks_quantity] - time[i]) << endl; // .. for debug
        elapsed_time += time[i + blocks_quantity] - time[i];
    }

    cout << endl << "total clock cycle used : " << elapsed_time << endl;

    cudaFree(d_time); // free preallocated gpu memory space for timing 

    cudaMemcpy(cpu_ptr_a_reduct.data(), gpu_ptr_a_reduct, memory_space_4vector, cudaMemcpyDeviceToHost);  // copy the resulting gpu stored vector, in cpu stored vector

    cout << "result on gpu computation : " << cpu_ptr_a_reduct[0] << endl; // espected 65536 because all elements are setted at value one, this result is computed in gpu
    cout << "result on cpu computation : " << accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0); // print on console the sum of vector elements, computed in cpu

    assert(cpu_ptr_a_reduct[0] == accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0)); // compare and check the rusults 

    cudaFree(gpu_ptr_a_original); // free preallocated gpu memory space
    cudaFree(gpu_ptr_a_reduct);

    cout << endl;
    printf(" ---> VECTOR SUM ACCELERATED - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    // SUM REDUCTION 2

    threads_quantity = 128;  // set number of threads to execute
    blocks_quantity = (int)ceil(vector_length / threads_quantity); // 32 blocks in this example

    cudaMalloc(&gpu_ptr_a_original, memory_space_4vector); // to allocate space in gpu memory 
    cudaMalloc(&gpu_ptr_a_reduct, memory_space_4vector);

    cudaMemcpy(gpu_ptr_a_original, cpu_ptr_a_original.data(), memory_space_4vector, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored vector in gpu allocated memory

    cudaMalloc(&d_time, sizeof(clock_t) * blocks_quantity * 2);  // allocate space in gpu ememory for time computation

    vectorutility_sumReduction2 <<<blocks_quantity, threads_quantity>>> (gpu_ptr_a_original, gpu_ptr_a_reduct, d_time); // execute this function 

    cudaMemcpy(time, d_time, sizeof(clock_t) * blocks_quantity * 2, cudaMemcpyDeviceToHost);

    vectorutility_sumReduction2 <<<1, threads_quantity>>> (gpu_ptr_a_reduct, gpu_ptr_a_reduct, d_time); // gpu have not a global block threads synchronization, so to syncronixe threads I recall this gpu fucntion again, but with only one block 

    cout << endl;

    elapsed_time = 0;

    for (int i = 0; i < blocks_quantity; i++)
    {
        // cout << i << "  " << (time[i + blocks_quantity] - time[i]) << endl; // .. for debug
        elapsed_time += time[i + blocks_quantity] - time[i];
    }

    cout << endl << "total clock cycle used : " << elapsed_time << endl;

    cudaFree(d_time); // free preallocated gpu memory space for timing 

    cudaMemcpy(cpu_ptr_a_reduct.data(), gpu_ptr_a_reduct, memory_space_4vector, cudaMemcpyDeviceToHost); // copy the resulting gpu stored vector, in cpu stored vector

    cout << "result on cpu computation : " << cpu_ptr_a_reduct[0] << endl; // espected 65536 because all elements are setted at value one, this result is computed in gpu
    cout << "result on cpu computation : " << accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0); // print on console the sum of vector elements, computed in cpu

    assert(cpu_ptr_a_reduct[0] == accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0)); // compare and check the rusults 

    cudaFree(gpu_ptr_a_original); // free preallocated gpu memory space
    cudaFree(gpu_ptr_a_reduct);

    cout << endl;
    printf(" ---> VECTOR SUM 2 - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    // SUM REDUCTION 2 ACCELERATED

    threads_quantity = 256;  // set number of threads to execute
    blocks_quantity = (int)ceil(vector_length / threads_quantity); // 64 blocks in this example

    cudaMalloc(&gpu_ptr_a_original, memory_space_4vector); // to allocate space in gpu memory 
    cudaMalloc(&gpu_ptr_a_reduct, memory_space_4vector);

    cudaMemcpy(gpu_ptr_a_original, cpu_ptr_a_original.data(), memory_space_4vector, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored vector in gpu allocated memory

    cudaMalloc(&d_time, sizeof(clock_t) * blocks_quantity * 2);  // allocate space in gpu ememory for time computation

    vectorutility_sumReduction2 << <blocks_quantity, threads_quantity >> > (gpu_ptr_a_original, gpu_ptr_a_reduct, d_time); // execute this function 

    cudaMemcpy(time, d_time, sizeof(clock_t) * blocks_quantity * 2, cudaMemcpyDeviceToHost);

    vectorutility_sumReduction2 << <1, threads_quantity >> > (gpu_ptr_a_reduct, gpu_ptr_a_reduct, d_time); // gpu have not a global block threads synchronization, so to syncronixe threads I recall this gpu fucntion again, but with only one block 

    cout << endl;

    elapsed_time = 0;

    for (int i = 0; i < blocks_quantity; i++)
    {
        // cout << i << "  " << (time[i + blocks_quantity] - time[i]) << endl; // .. for debug
        elapsed_time += time[i + blocks_quantity] - time[i];
    }

    cout << endl << "total clock cycle used : " << elapsed_time << endl;

    cudaFree(d_time); // free preallocated gpu memory space for timing 

    cudaMemcpy(cpu_ptr_a_reduct.data(), gpu_ptr_a_reduct, memory_space_4vector, cudaMemcpyDeviceToHost); // copy the resulting gpu stored vector, in cpu stored vector

    cout << "result on cpu computation : " << cpu_ptr_a_reduct[0] << endl; // espected 65536 because all elements are setted at value one, this result is computed in gpu
    cout << "result on cpu computation : " << accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0); // print on console the sum of vector elements, computed in cpu

    assert(cpu_ptr_a_reduct[0] == accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0)); // compare and check the rusults 

    cudaFree(gpu_ptr_a_original); // free preallocated gpu memory space
    cudaFree(gpu_ptr_a_reduct);

    cout << endl;
    printf(" ---> VECTOR SUM 2 ACCELERATED - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    // SUM REDUCTION 3

    threads_quantity = 128;  // set number of threads to execute
    blocks_quantity = (int)ceil(vector_length / threads_quantity); // 32 blocks in this example

    cudaMalloc(&gpu_ptr_a_original, memory_space_4vector); // to allocate space in gpu memory 
    cudaMalloc(&gpu_ptr_a_reduct, memory_space_4vector);

    cudaMemcpy(gpu_ptr_a_original, cpu_ptr_a_original.data(), memory_space_4vector, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored vector in gpu allocated memory

    cudaMalloc(&d_time, sizeof(clock_t) * blocks_quantity * 2); // allocate space in gpu ememory for time computation

    vectorutility_sumReduction3 <<<blocks_quantity, threads_quantity>>> (gpu_ptr_a_original, gpu_ptr_a_reduct, d_time); // execute this function 

    cudaMemcpy(time, d_time, sizeof(clock_t) * blocks_quantity * 2, cudaMemcpyDeviceToHost);

    vectorutility_sumReduction3 <<<1, threads_quantity>>> (gpu_ptr_a_reduct, gpu_ptr_a_reduct, d_time); // gpu have not a global block threads synchronization, so to syncronixe threads I recall this gpu fucntion again, but with only one block 

    cout << endl;

    elapsed_time = 0;

    for (int i = 0; i < blocks_quantity; i++)
    {
        // cout << i << "  " << (time[i + blocks_quantity] - time[i]) << endl; // .. for debug
        elapsed_time += time[i + blocks_quantity] - time[i];
    }

    cout << endl << "total clock cycle used : " << elapsed_time << endl;

    cudaFree(d_time); // free preallocated gpu memory space for timing 

    cudaMemcpy(cpu_ptr_a_reduct.data(), gpu_ptr_a_reduct, memory_space_4vector, cudaMemcpyDeviceToHost); // copy the resulting gpu stored vector, in cpu stored vector

    cout << "result on gpu computation : " << cpu_ptr_a_reduct[0] << endl; // espected 65536 because all elements are setted at value one, this result is computed in gpu
    cout << "result on cpu computation : " << accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0); // print on console the sum of vector elements, computed in cpu

    assert(cpu_ptr_a_reduct[0] == accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0)); // compare and check the rusults 

    cudaFree(gpu_ptr_a_original); // free preallocated gpu memory space
    cudaFree(gpu_ptr_a_reduct);

    cout << endl;
    printf(" ---> VECTOR SUM 3- OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);

    // SUM REDUCTION 3 ACCELERATED

    threads_quantity = 256;  // set number of threads to execute
    blocks_quantity = (int)ceil(vector_length / threads_quantity); // 64 blocks in this example

    cudaMalloc(&gpu_ptr_a_original, memory_space_4vector); // to allocate space in gpu memory 
    cudaMalloc(&gpu_ptr_a_reduct, memory_space_4vector);

    cudaMemcpy(gpu_ptr_a_original, cpu_ptr_a_original.data(), memory_space_4vector, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored vector in gpu allocated memory

    cudaMalloc(&d_time, sizeof(clock_t) * blocks_quantity * 2); // allocate space in gpu ememory for time computation

    vectorutility_sumReduction3 << <blocks_quantity, threads_quantity >> > (gpu_ptr_a_original, gpu_ptr_a_reduct, d_time); // execute this function 

    cudaMemcpy(time, d_time, sizeof(clock_t) * blocks_quantity * 2, cudaMemcpyDeviceToHost);

    vectorutility_sumReduction3 << <1, threads_quantity >> > (gpu_ptr_a_reduct, gpu_ptr_a_reduct, d_time); // gpu have not a global block threads synchronization, so to syncronixe threads I recall this gpu fucntion again, but with only one block 

    cout << endl;

    elapsed_time = 0;

    for (int i = 0; i < blocks_quantity; i++)
    {
        // cout << i << "  " << (time[i + blocks_quantity] - time[i]) << endl; // .. for debug
        elapsed_time += time[i + blocks_quantity] - time[i];
    }

    cout << endl << "total clock cycle used : " << elapsed_time << endl;

    cudaFree(d_time); // free preallocated gpu memory space for timing 

    cudaMemcpy(cpu_ptr_a_reduct.data(), gpu_ptr_a_reduct, memory_space_4vector, cudaMemcpyDeviceToHost); // copy the resulting gpu stored vector, in cpu stored vector

    cout << "result on gpu computation : " << cpu_ptr_a_reduct[0] << endl; // espected 65536 because all elements are setted at value one, this result is computed in gpu
    cout << "result on cpu computation : " << accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0); // print on console the sum of vector elements, computed in cpu

    assert(cpu_ptr_a_reduct[0] == accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0)); // compare and check the rusults 

    cudaFree(gpu_ptr_a_original); // free preallocated gpu memory space
    cudaFree(gpu_ptr_a_reduct);

    cout << endl;
    printf(" ---> VECTOR SUM 3 ACCELERATED - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    // SUM REDUCTION 4

    threads_quantity = 128;  // set number of threads to execute
    blocks_quantity = (int)ceil(vector_length / threads_quantity); // 32 blocks in this example

    cudaMalloc(&gpu_ptr_a_original, memory_space_4vector); // to allocate space in gpu memory 
    cudaMalloc(&gpu_ptr_a_reduct, memory_space_4vector);

    cudaMemcpy(gpu_ptr_a_original, cpu_ptr_a_original.data(), memory_space_4vector, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored vector in gpu allocated memory

    cudaMalloc(&d_time, sizeof(clock_t) * blocks_quantity * 2); // allocate space in gpu ememory for time computation

    vectorutility_sumReduction4 <<<blocks_quantity, threads_quantity>>> (gpu_ptr_a_original, gpu_ptr_a_reduct, d_time); // execute this function 

    cudaMemcpy(time, d_time, sizeof(clock_t) * blocks_quantity * 2, cudaMemcpyDeviceToHost); 

    vectorutility_sumReduction4 <<<1, threads_quantity>>> (gpu_ptr_a_reduct, gpu_ptr_a_reduct, d_time); // gpu have not a global block threads synchronization, so to syncronixe threads I recall this gpu fucntion again, but with only one block 

    cout << endl;

    elapsed_time = 0;

    for (int i = 0; i < blocks_quantity; i++)
    {
        // cout << i << "  " << (time[i + blocks_quantity] - time[i]) << endl; // -- for debug
        elapsed_time += time[i + blocks_quantity] - time[i];
    }

    cout << endl << "total clock cycle used : " << elapsed_time << endl;

    cudaFree(d_time); // free preallocated gpu memory space for timing 

    cudaMemcpy(cpu_ptr_a_reduct.data(), gpu_ptr_a_reduct, memory_space_4vector, cudaMemcpyDeviceToHost); // copy the resulting gpu stored vector, in cpu stored vector

    cout << "result on gpu computation : " << cpu_ptr_a_reduct[0] << endl; // espected 65536 because all elements are setted at value one, this result is computed in gpu
    cout << "result on cpu computation : " << accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0); // print on console the sum of vector elements, computed in cpu 

    assert(cpu_ptr_a_reduct[0] == accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0)); // in first position (position zero) of cpu memory stored "reducted vector" is assigned the result of addiction of all original vector elements 

    cudaFree(gpu_ptr_a_original); // free preallocated gpu memory space
    cudaFree(gpu_ptr_a_reduct);

    cout << endl;
    printf(" ---> VECTOR SUM 4 - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    // SUM REDUCTION 4 ACCELERATED

    threads_quantity = 256;  // set number of threads to execute
    blocks_quantity = (int)ceil(vector_length / threads_quantity); // 64 blocks in this example

    cudaMalloc(&gpu_ptr_a_original, memory_space_4vector); // to allocate space in gpu memory 
    cudaMalloc(&gpu_ptr_a_reduct, memory_space_4vector);

    cudaMemcpy(gpu_ptr_a_original, cpu_ptr_a_original.data(), memory_space_4vector, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored vector in gpu allocated memory

    cudaMalloc(&d_time, sizeof(clock_t) * blocks_quantity * 2); // allocate space in gpu ememory for time computation

    vectorutility_sumReduction4 << <blocks_quantity, threads_quantity >> > (gpu_ptr_a_original, gpu_ptr_a_reduct, d_time); // execute this function 

    cudaMemcpy(time, d_time, sizeof(clock_t) * blocks_quantity * 2, cudaMemcpyDeviceToHost);

    vectorutility_sumReduction4 << <1, threads_quantity >> > (gpu_ptr_a_reduct, gpu_ptr_a_reduct, d_time); // gpu have not a global block threads synchronization, so to syncronixe threads I recall this gpu fucntion again, but with only one block 

    cout << endl;

    elapsed_time = 0;

    for (int i = 0; i < blocks_quantity; i++)
    {
        // cout << i << "  " << (time[i + blocks_quantity] - time[i]) << endl; // -- for debug
        elapsed_time += time[i + blocks_quantity] - time[i];
    }

    cout << endl << "total clock cycle used : " << elapsed_time << endl;

    cudaFree(d_time); // free preallocated gpu memory space for timing 

    cudaMemcpy(cpu_ptr_a_reduct.data(), gpu_ptr_a_reduct, memory_space_4vector, cudaMemcpyDeviceToHost); // copy the resulting gpu stored vector, in cpu stored vector

    cout << "result on gpu computation : " << cpu_ptr_a_reduct[0] << endl; // espected 65536 because all elements are setted at value one, this result is computed in gpu
    cout << "result on cpu computation : " << accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0); // print on console the sum of vector elements, computed in cpu 

    assert(cpu_ptr_a_reduct[0] == accumulate(begin(cpu_ptr_a_original), end(cpu_ptr_a_original), 0)); // in first position (position zero) of cpu memory stored "reducted vector" is assigned the result of addiction of all original vector elements 

    cudaFree(gpu_ptr_a_original); // free preallocated gpu memory space
    cudaFree(gpu_ptr_a_reduct);

    cout << endl;
    printf(" ---> VECTOR SUM 4 ACCELERATED - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    // MATRIX MULTIPLICATION

    int matrix_length = 1 << 5; // 32 in decimal

    size_t memory_space_for_matrix = matrix_length * matrix_length * sizeof(int);  // for matrix (squared matrix, in this example), mem. space to allocate is column number * row number * memory ocuupied by variable type 

    threads_quantity = 8; // set number of threads to execute in one block
    blocks_quantity = ((matrix_length + threads_quantity - 1) / threads_quantity);

    dim3 threads_In_OneBlock(threads_quantity, threads_quantity); // "dim3" type is a sort of container included in cuda libraries, here declared and initialized with only two elements 
    dim3 blocks(blocks_quantity, blocks_quantity); // matrix of blocks

    cpu_ptr_a = (int*)malloc(memory_space_for_matrix); // to allocate space in cpu memory (heap) (variable of "pointer type" are previously declared)
    cpu_ptr_b= (int*)malloc(memory_space_for_matrix);
    cpu_ptr_c = (int*)malloc(memory_space_for_matrix);

    cudaMalloc(&gpu_ptr_a, memory_space_for_matrix); // to allocate space in gpu memory (variable of "pointer type" are previously declared)
    cudaMalloc(&gpu_ptr_b, memory_space_for_matrix);
    cudaMalloc(&gpu_ptr_c, memory_space_for_matrix);

    matrixutility_randomInit(cpu_ptr_a, matrix_length); // initialize the matrix with pseudo random values
    matrixutility_randomInit(cpu_ptr_b, matrix_length);

    cudaMemcpy(gpu_ptr_a, cpu_ptr_a, memory_space_for_matrix, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored matrix in gpu allocated memory
    cudaMemcpy(gpu_ptr_b, cpu_ptr_b, memory_space_for_matrix, cudaMemcpyHostToDevice);

    matrixutility_mul <<<blocks, threads_In_OneBlock>>> (gpu_ptr_a, gpu_ptr_b, gpu_ptr_c, matrix_length); // gpu processing, in this time parameters are of "dim3" type

    cudaDeviceSynchronize(); // wait the end of all parallelized operation

    cudaMemcpy(cpu_ptr_c, gpu_ptr_c, memory_space_for_matrix, cudaMemcpyDeviceToHost); // copy the resulting gpu stored vector, in cpu stored vector

    matrixutility_errorCheck(cpu_ptr_a, cpu_ptr_b, cpu_ptr_c, matrix_length); // ...to verify the result by cpu procesing

    cudaFree(gpu_ptr_a); // free preallocated gpu memory space
    cudaFree(gpu_ptr_b);
    cudaFree(gpu_ptr_c);

    free(cpu_ptr_a); // free preallocated cpu memory space (heap)
    free(cpu_ptr_b);
    free(cpu_ptr_c);

    cout << endl;
    printf(" ---> MUL MATRIX - OK ! <--- "); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    // TILED MATRIX MULTIPLICATION

    matrix_length = (1 << 5); // 100000b = 32d --> matrix dimension is 32 x 32 = 1024 elements

    memory_space_for_matrix = matrix_length * matrix_length * sizeof(int); // for matrix (squared matrix, in this example), mem. space to allocate is column number * row number * memory ocuupied by variable type 

    threads_quantity = (matrix_length / 4); // set number of threads to execute in one block: here is set as 1/4 of global matrix length, the same quantity of tile length
    blocks_quantity = ((matrix_length + threads_quantity - 1) / threads_quantity);
    int tile_length = threads_quantity; 

    dim3 threads_In_OneBlock_2(threads_quantity, threads_quantity); // "dim3" type is a sort of container included in cuda libraries, here declared and initialized with only two elements 
    dim3 blocks_2(blocks_quantity, blocks_quantity); // new definition (differentiatet with "_2" suffix) and initialization for this dim3 type variable
    
    cpu_ptr_a = (int*)malloc(memory_space_for_matrix); // to allocate space in cpu memory (heap) (variable of "pointer type" are previously declared)
    cpu_ptr_b = (int*)malloc(memory_space_for_matrix);
    cpu_ptr_c = (int*)malloc(memory_space_for_matrix);

    cudaMalloc(&gpu_ptr_a, memory_space_for_matrix); // to allocate space in gpu memory (variable of "pointer type" are previously declared)
    cudaMalloc(&gpu_ptr_b, memory_space_for_matrix);
    cudaMalloc(&gpu_ptr_c, memory_space_for_matrix);

    //matrixutility_1_or_0_init(cpu_ptr_a, matrix_length); // initialize the matrix with one or zero values
    //matrixutility_1_or_0_init(cpu_ptr_b, matrix_length);

    matrixutility_randomInit(cpu_ptr_a, matrix_length); // initialize the matrix with pseudo random values
    matrixutility_randomInit(cpu_ptr_b, matrix_length);

    cudaMemcpy(gpu_ptr_a, cpu_ptr_a, memory_space_for_matrix, cudaMemcpyHostToDevice); // copy the value of elements of cpu stored matrix in gpu allocated memory
    cudaMemcpy(gpu_ptr_b, cpu_ptr_b, memory_space_for_matrix, cudaMemcpyHostToDevice);

    matrixutility_mul2 <<<blocks_2, threads_In_OneBlock_2>>> (gpu_ptr_a, gpu_ptr_b, gpu_ptr_c, matrix_length, tile_length); // gpu processing 

    cudaMemcpy(cpu_ptr_c, gpu_ptr_c, memory_space_for_matrix, cudaMemcpyDeviceToHost); // copy the resulting gpu stored vector, in cpu stored vector

    matrixutility_errorCheck(cpu_ptr_a, cpu_ptr_b, cpu_ptr_c, matrix_length); // ...to verify the result by cpu procesing

    cudaFree(gpu_ptr_a); // free preallocated gpu memory space
    cudaFree(gpu_ptr_b);
    cudaFree(gpu_ptr_c);

    free(cpu_ptr_a); // free preallocated cpu memory space (heap)
    free(cpu_ptr_b);
    free(cpu_ptr_c);

    cout << endl;
    printf(" ---> MUL TILED MATRIX - OK ! <---"); // print success message in console if there aren't errors 
    cout << endl;

    Sleep(2000);


    return 0;
}
