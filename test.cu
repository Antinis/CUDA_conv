#include <cuda.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <mma.h>

__global__ void kernel(unsigned char *a, unsigned char *w, unsigned char *b)
{
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, unsigned char, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, unsigned char, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0);

    nvcuda::wmma::load_matrix_sync(a_frag, a, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, w, 16);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncthreads();

    int res[256]={0};
    nvcuda::wmma::store_matrix_sync(res, c_frag, 16, nvcuda::wmma::mem_row_major);

    __syncthreads();

    if(threadIdx.x==0)
    {
      for(int i=0; i<16; i++)
      {
          for(int j=0; j<16; j++)
              printf("%d ", res[i*16+j]);
          printf("\n");
      }
      printf("\n");
    }

    

    for(int i=0; i<256; i++)
        b[i]=res[i];
    
        if(threadIdx.x==0)
        {
          for(int i=0; i<16; i++)
          {
              for(int j=0; j<16; j++)
                  printf("%d ", res[i*16+j]);
              printf("\n");
          }
          printf("\n");
        }

    return;
}

int main()
{
    uint8_t a[128];
    uint8_t w[128];
    uint8_t b[128];
    for(int i=0; i<128; i++)
        a[i]=1;
    for(int i=0; i<128; i++)
        w[i]=1;
    
    uint8_t *a_kernel, *w_kernel, *b_kernel;
    cudaMalloc(&a_kernel, 128*sizeof(uint8_t));
    cudaMalloc(&w_kernel, 128*sizeof(uint8_t));
    cudaMalloc(&b_kernel, 128*sizeof(uint8_t));
    cudaMemcpy(a_kernel, a, 128*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(w_kernel, w, 128*sizeof(uint8_t), cudaMemcpyHostToDevice);

    kernel<<<1, 32>>>(a_kernel, w_kernel, b_kernel);
    cudaMemcpy(b, b_kernel, 128*sizeof(uint8_t), cudaMemcpyDeviceToHost);

    int res[128]={0};
    for(int i=0; i<128; i++)
        res[i]=b[i];

    for(int i=0; i<16; i++)
    {
        for(int j=0; j<16; j++)
            std::cout<<res[i]<<" ";
        std::cout<<std::endl;
    }
    
    return 0;
}

// using namespace nvcuda;

// __global__ void wmma_ker(half *a, half *b, float *c) {
//    // Declare the fragments
//    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
//    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
//    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

//    // Initialize the output to zero
//    wmma::fill_fragment(c_frag, 0.0f);

//    // Load the inputs
//    wmma::load_matrix_sync(a_frag, a, 16);
//    wmma::load_matrix_sync(b_frag, b, 16);

//    // Perform the matrix multiplication
//    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

//    // Store the output
//    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);

//   //  // 输出
//   //  for(int i=0; i<16; i++)
//   //   {
//   //       for(int j=0; j<16; j++)
//   //           printf("%f ", __half2float(a[i*16+j]));
//   //       printf("\n");
//   //   }
//   //   printf("\n");

//   //   for(int i=0; i<16; i++)
//   //   {
//   //       for(int j=0; j<16; j++)
//   //           printf("%f ", __half2float(b[i*16+j]));
//   //       printf("\n");
//   //   }
//   //   printf("\n");

//   //   for(int i=0; i<16; i++)
//   //   {
//   //       for(int j=0; j<16; j++)
//   //           printf("%f ", c[i*16+j]);
//   //       printf("\n");
//   //   }
//   //   printf("\n");
// }

// int main()
// {
//     uint8_t a[256];
//     uint8_t w[256];
//     int b[256];
//     for(int i=0; i<256; i++)
//         a[i]=1;
//     for(int i=0; i<256; i++)
//         w[i]=1;
    
//     half *a_kernel, *w_kernel;
//     float *b_kernel;
//     cudaMalloc(&a_kernel, 256*sizeof(uint8_t));
//     cudaMalloc(&w_kernel, 256*sizeof(uint8_t));
//     cudaMalloc(&b_kernel, 256*sizeof(int));
//     cudaMemcpy(a_kernel, a, 256*sizeof(uint8_t), cudaMemcpyHostToDevice);
//     cudaMemcpy(w_kernel, w, 256*sizeof(uint8_t), cudaMemcpyHostToDevice);

//     wmma_ker<<<1, 16>>>(a_kernel, w_kernel, b_kernel);
//     cudaDeviceSynchronize();
//     cudaMemcpy(b, b_kernel, 256*sizeof(int), cudaMemcpyDeviceToHost);

//     // 输出
//     for(int i=0; i<16; i++)
//     {
//         for(int j=0; j<16; j++)
//             // printf("%f ", __half2float(a[i*16+j]));
//             printf("%f ", b[i*16+j]);
//         printf("\n");
//     }
    
//     return 0;
// }