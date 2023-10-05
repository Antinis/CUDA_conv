#include <cuda.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <time.h>

const int alignment = 32; // 32 byte alignment
const int size = 100;
const int kernel = 3;  // odd
const int batch_size = 128;
const int in_channel = 128;
const int out_channel = 128;

#define InitRandom()                         \
  std::random_device r;                      \
  std::default_random_engine generator; \
  std::uniform_int_distribution<> distribution(0, 255);

#define a(_n, _x, _y, _c) a[(_n) * size * size * in_channel + (_x) * size * in_channel + (_y) * in_channel + (_c)]
#define w(_x, _y, _ci, _co) w[(_x) * kernel * in_channel * out_channel + (_y) * in_channel * out_channel + (_ci) * out_channel + (_co)]
#define b(_n, _x, _y, _c) b[(_n) * size * size * out_channel + (_x) * size * out_channel + (_y) * out_channel + (_c)]

/// \brief Generate [N, H, W, C] input tensor and [H, W, I, O] kernel tensor.
void Generate(uint8_t *const a, uint8_t *const w) {
#pragma omp parallel for
  // Batch dimension.
  for (int s = 0; s < batch_size; ++s) {
      InitRandom();
      // Height dimension.
      for (int i = 0; i < size; ++i)
        // Width dimension.
        for (int j = 0; j < size; ++j) {
          const int channel_lower = s * size * size * in_channel
                                  + i * size * in_channel
                                  + j * in_channel;
          const int channel_upper = channel_lower + in_channel; 
          // Channel dimension.
          for (int c = channel_lower; c < channel_upper; ++c)
            a[c] = distribution(generator);
        }
  }
#pragma omp parallel for
  for (int i = 0; i < kernel; ++i) {
    InitRandom();
    for (int j = 0; j < kernel; ++j) 
      for (int CI = 0; CI < in_channel; ++CI) {
        const int channel_lower = i * kernel * in_channel * out_channel
                                + j * in_channel * out_channel
                                + CI * out_channel;
        const int channel_upper = channel_lower + out_channel;
        for (int CO = channel_lower; CO < channel_upper; ++CO) 
          w[CO] = distribution(generator);
      }
  }
}

void conv2d_cpu_kernel(const uint8_t *__restrict__ a, 
                       const uint8_t *__restrict__ w, 
                       uint8_t *__restrict__ b) {
#pragma omp parallel for
  for (int s = 0; s < batch_size; ++s) {
    size_t output_bytes = ((out_channel * sizeof(uint8_t)) + (size_t)alignment - 1) & ~((size_t)alignment -1); 
    uint8_t *packedB = static_cast<uint8_t *>(malloc(output_bytes));

    size_t input_bytes = ((kernel * kernel * in_channel * sizeof(uint8_t)) + (size_t)alignment - 1) & ~((size_t)alignment - 1);
    uint8_t *packedA = static_cast<uint8_t *>(malloc(input_bytes));

    for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j) {
        // Collected needed input data,
        // Start from A[s, i - kernel / 2, j - kernel / 2, 0].
        int x = i - kernel / 2;
        int y = j - kernel / 2;
        int input_index = s * size * size * in_channel
                        + x * size * in_channel
                        + y * in_channel;
        memset(packedA, 0, input_bytes);
        int A_buffer_index = 0;
        for (int kh = 0; kh < kernel; ++kh) {
          for (int kw = 0; kw < kernel; ++ kw) {
            if (!(x < 0 || x >= size || y < 0 || y >= size)) {
              memcpy(packedA + A_buffer_index, a + input_index, in_channel * sizeof(uint8_t));
            }
            else {
              memset(packedA + A_buffer_index, 0, in_channel * sizeof(uint8_t));
            }
            y++;
            A_buffer_index += in_channel;
            input_index += in_channel;
          }
          x++;
          y -= kernel;
          input_index = input_index - kernel * in_channel + size * in_channel;
        }

        // Start from B[s, i, j, 0]
        int output_index = s * size * size * out_channel 
                         + i * size * out_channel 
                         + j * out_channel;                 
        memset(packedB, 0, output_bytes);

        // Start from W[0, 0, 0, 0]
        int kernel_index = 0;
        A_buffer_index = 0;
        // Convolution 2D computation.
        // iterate over each in_channel of input tensor,
        // and accumulate contribution to output tensor.
        for (int N = 0; N < kernel * kernel; ++N) {
          for (int CI = 0; CI < in_channel; ++CI) {
            for (int CO = 0; CO < out_channel; ++CO) {
              packedB[CO] +=  packedA[A_buffer_index] * w[kernel_index];
              kernel_index++; // move to next output channel.
            }
            A_buffer_index++;
          }
        }
        memcpy(b + output_index, packedB, sizeof(uint8_t) * out_channel);
      }
    free(packedA);
    free(packedB);
  }
}

time_t Check(const uint8_t *const a, const uint8_t *const w, uint8_t *const b) {
  auto b_std = new uint8_t[batch_size * size * size * out_channel];

  std::cout << "Conv2d CPU Kernel Start... \n";
  time_t cpu_time_start, cpu_time_end;
  cpu_time_start=clock();
  conv2d_cpu_kernel(a, w, b_std);
  cpu_time_end=clock();
  std::cout << "Checking Results... \n";
  time_t cpu_time=(double)((cpu_time_end-cpu_time_start)/CLOCKS_PER_SEC);

  size_t N = batch_size * size * size * out_channel;
  for (size_t i = 0; i < N; ++i) {
    if (b[i] != b_std[i]) {
      std::cout << "\x1b[31m"
                   "Wrong Answer"
                   "\x1b[0m"
                   " at "
                << i << std::endl;
      std::cout << "expected " << (int)b_std[i] << " but found " << (int)b[i]
                << std::endl;
      delete[] b_std;
      return cpu_time;
    }
  }
  std::cout << "\x1b[32m"
               "Correct"
               "\x1b[0m"
            << std::endl;

  delete[] b_std;

  return cpu_time;
}

/// \brief Do Conv2d with NHWC Input with HWIO Kernel, and NHWC output 
__global__ void conv2d_cuda_kernel(const uint8_t *a, 
                                   const uint8_t *w, 
                                   uint8_t *b) 
{
    //i是横向编号
    //j是纵向编号
    const int batch_id=blockIdx.x;  //不会变
    const int x=blockIdx.y;         //不会变
    const int y=blockIdx.z;         //不会变
    const int out_channel_id=threadIdx.x;   //只有这个会变

    //遍历卷积核每一个像素点
    uint8_t conv=0;
    for(int in_channel_id=0; in_channel_id<in_channel; in_channel_id++)
    {
        int kx = x - kernel / 2;
        int ky = y - kernel / 2; //待卷积位置的左上角坐标
        for(int k=0; k<kernel; k++)
        {
            for(int l=0; l<kernel; l++)
            {
                if(!(kx<0 || kx>=size || ky<0 || ky>=size))
                    conv+=a(batch_id, kx, ky, in_channel_id)*w(k, l, in_channel_id, out_channel_id);
                ky++;
            }
            kx++;
            ky-=kernel;
        }
    }
    // Write back to b.
    b(batch_id, x, y, out_channel_id) = conv;

    return;
}

// naive and shit
// only for testing correctness and precision
void conv_cuda(const uint8_t *const a, const uint8_t *const w, uint8_t *const b,
               cudaEvent_t *start_e, cudaEvent_t *stop_e) 
{
    //建立显存上的存储空间并拷贝变量
    uint8_t *a_kernel, *w_kernel, *b_kernel;
    //size是长宽尺寸，是正方形
    cudaMalloc(&a_kernel, batch_size * size * size * in_channel * sizeof(uint8_t));
    cudaMemcpy(a_kernel, a, batch_size * size * size * in_channel * sizeof(uint8_t),
                cudaMemcpyHostToDevice);
    cudaMalloc(&w_kernel, kernel * kernel * in_channel * out_channel * sizeof(uint8_t));
    cudaMemcpy(w_kernel, w, kernel * kernel * in_channel * out_channel * sizeof(uint8_t),
                cudaMemcpyHostToDevice);
    cudaMalloc(&b_kernel, batch_size * size * size * out_channel * sizeof(uint8_t));
    // Start Timer.
    cudaEventRecord(*start_e);
    // Run Conv2d Kernel,
    // Timer for computation cuda kernel.
    //规定计算器件的数量和维度尺寸（三维）,在该例中第三维（长度维度）自动补1
    //gird是高宽都是100个盒子（size），长度方向盒子数1的区域
    //block是高宽都是16个线程（block_size），长度方向线程数1的盒子

    //这一版并行化了batch、CI和CO维度

    dim3 grid(batch_size, size, size);  //盒子间并行化
    dim3 block(out_channel); //10个盒子并行一张图片的处理
    //负责计算的核函数
    conv2d_cuda_kernel<<<grid, block>>>(a_kernel, w_kernel, b_kernel);
    cudaDeviceSynchronize();
    // Stop Timer
    cudaEventRecord(*stop_e);
    cudaEventSynchronize(*stop_e);

    cudaMemcpy(b, b_kernel, batch_size * size * size * out_channel * sizeof(uint8_t),
                cudaMemcpyDeviceToHost);
    cudaFree(a_kernel);
    cudaFree(w_kernel);
    cudaFree(b_kernel);
}

int main() {
    //a是原图像张量，w是卷积核张量，b是卷积结果张量
    auto a = new uint8_t[batch_size * size * size * in_channel];
    auto w = new uint8_t[kernel * kernel * in_channel * out_channel];
    auto b = new uint8_t[batch_size * size * size * out_channel];
    //随机生成a和w
    std::cout << "Generating input and kernel tensor... \n";
    Generate(a, w);

    //记录时间的变量
    cudaEvent_t start_e, stop_e;
    cudaEventCreate(&start_e);
    cudaEventCreate(&stop_e);

    // Conv(a, w, b)，用CUDA;
    std::cout << "Conv2d Cuda Kernel Start... \n";
    conv_cuda(a, w, b, &start_e, &stop_e);

    //检查正确性，check函数里使用CPU循环操作卷积
    std::cout << "Verifying... \n";
    time_t cpu_time=Check(a, w, b);
    //CUDA计算时间，毫秒
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_e, stop_e);
    std::cout <<"CUDA time: " << milliseconds << " ms" << std::endl;
    std::cout << "accelerate ratio: " << cpu_time*1000/milliseconds << std::endl;
    //销毁变量 
    cudaEventDestroy(start_e);
    cudaEventDestroy(stop_e);

    // Output(a, w, b);
    delete[] a;
    delete[] w;
    delete[] b;
    return 0;
}