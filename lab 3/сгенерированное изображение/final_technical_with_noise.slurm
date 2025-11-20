#!/bin/bash
#SBATCH --job-name=technical_with_noise
#SBATCH --partition=gpuserv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=output_technical_with_noise_%j.log

echo "=== Final Technical with Noise ==="
module load nvidia/cuda-11.7

cat > final_technical_with_noise.cu << 'EOF'
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <chrono>
#include <png.h>
#include <thread>
#include <future>
#include <random>

struct Image {
    int width, height, channels;
    std::vector<unsigned char> data;
    Image() : width(0), height(0), channels(0) {}
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(w * h * c);
    }
};

// Функция для расчета количества ядер на основе Compute Capability
int get_cores_per_sm(int major, int minor) {
    struct SMConfig {
        int major, minor, cores;
    };
    
    SMConfig configs[] = {
        {3, 0, 192}, {3, 5, 192}, {3, 7, 192},
        {5, 0, 128}, {5, 2, 128}, {5, 3, 128},
        {6, 0, 64}, {6, 1, 128}, {6, 2, 128},
        {7, 0, 64}, {7, 2, 64}, {7, 5, 64},
        {8, 0, 64}, {8, 6, 128},
        {8, 9, 128}, {9, 0, 128}
    };
    
    for (const auto& config : configs) {
        if (config.major == major && config.minor == minor) {
            return config.cores;
        }
    }
    return 128;
}

// Функция для вывода технических характеристик GPU
void print_gpu_specs() {
    std::cout << "\n=== GPU TECHNICAL SPECIFICATIONS ===" << std::endl;
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        int cores_per_sm = get_cores_per_sm(prop.major, prop.minor);
        int total_cores = prop.multiProcessorCount * cores_per_sm;
        
        std::cout << "--- GPU " << i << " ---" << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "Memory Clock: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "Memory Bandwidth: " << 
            (2.0 * prop.memoryClockRate * 1000 * (prop.memoryBusWidth / 8)) / 1e9 << " GB/s" << std::endl;
        std::cout << "L2 Cache: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "Cores per SM: " << cores_per_sm << std::endl;
        std::cout << "Total Cores: " << total_cores << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads per MP: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Concurrent Kernels: " << prop.concurrentKernels << std::endl;
        std::cout << "PCI Bus ID: " << prop.pciBusID << std::endl;
        std::cout << "PCI Device ID: " << prop.pciDeviceID << std::endl;
        
        // Проверяем текущую загрузку
        size_t free, total;
        cudaSetDevice(i);
        cudaMemGetInfo(&free, &total);
        std::cout << "Memory Usage: " << (total - free) / (1024*1024) << " MB / " 
                  << total / (1024*1024) << " MB" << std::endl;
        std::cout << "Memory Free: " << (free * 100) / total << "%" << std::endl;
        
        // Проверяем режим ECC
        std::cout << "ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
    }
    std::cout << "=====================================" << std::endl;
}

// CUDA ядра (без изменений)
__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float kernel[3][3] = {
        {1.0f/16, 2.0f/16, 1.0f/16},
        {2.0f/16, 4.0f/16, 2.0f/16},
        {1.0f/16, 2.0f/16, 1.0f/16}
    };
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int idx = (ny * width + nx) * channels + c;
                    sum += input[idx] * kernel[ky+1][kx+1];
                }
            }
        }
        int idx = (y * width + x) * channels + c;
        output[idx] = static_cast<unsigned char>(fminf(fmaxf(sum, 0.0f), 255.0f));
    }
}

__global__ void sobel_edge_kernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 1 || x >= width-1 || y < 1 || y >= height-1) return;
    
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for (int c = 0; c < channels; c++) {
        int gx = 0, gy = 0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int idx = ((y + ky) * width + (x + kx)) * channels + c;
                int pixel = input[idx];
                gx += pixel * sobel_x[ky+1][kx+1];
                gy += pixel * sobel_y[ky+1][kx+1];
            }
        }
        int magnitude = static_cast<int>(sqrtf(gx*gx + gy*gy));
        magnitude = (magnitude < 0) ? 0 : ((magnitude > 255) ? 255 : magnitude);
        int inverted_magnitude = 255 - magnitude;
        int idx = (y * width + x) * channels + c;
        output[idx] = static_cast<unsigned char>(inverted_magnitude);
    }
}

__global__ void median_filter_kernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 1 || x >= width-1 || y < 1 || y >= height-1) return;
    
    for (int c = 0; c < channels; c++) {
        unsigned char window[9];
        int count = 0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int idx = ((y + ky) * width + (x + kx)) * channels + c;
                window[count++] = input[idx];
            }
        }
        
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8 - i; j++) {
                if (window[j] > window[j+1]) {
                    unsigned char temp = window[j];
                    window[j] = window[j+1];
                    window[j+1] = temp;
                }
            }
        }
        int idx = (y * width + x) * channels + c;
        output[idx] = window[4];
    }
}

// Функция для создания тестового изображения С ШУМОМ
void create_test_image_with_noise(Image& image) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    std::uniform_int_distribution<> noise_dis(0, 19); // 5% вероятность шума
    
    std::cout << "Creating test image WITH NOISE (610x381)" << std::endl;
    
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            int idx = (y * image.width + x) * 3;
            
            // Создаем цветной градиент
            image.data[idx] = (x * 255) / image.width;     // Red
            image.data[idx + 1] = (y * 255) / image.height; // Green  
            image.data[idx + 2] = 150;                     // Blue
            
            // ДОБАВЛЯЕМ ШУМ (как в оригинальной single-GPU версии)
            if (noise_dis(gen) == 0) { // 5% вероятность шума
                image.data[idx] = dis(gen);     // Случайный красный
                image.data[idx + 1] = dis(gen); // Случайный зеленый
                image.data[idx + 2] = dis(gen); // Случайный синий
            }
        }
    }
    std::cout << "Test image created with random noise (5% pixels affected)" << std::endl;
}

// Функция для обработки части изображения на конкретном GPU
float process_part_parallel(const Image& input, Image& output, const std::string& filter_type, 
                           int gpu_id, int start_row, int end_row) {
    cudaSetDevice(gpu_id);
    
    int height = end_row - start_row;
    int width = input.width;
    int channels = input.channels;
    
    // Создаем временный буфер для этой части с расширенными границами
    int extended_height = height + 2;
    std::vector<unsigned char> extended_input(width * extended_height * channels);
    std::vector<unsigned char> extended_output(width * extended_height * channels);
    
    // Заполняем расширенный буфер
    for (int y = 0; y < extended_height; y++) {
        int src_y = start_row + y - 1;
        if (src_y < 0) src_y = 0;
        if (src_y >= input.height) src_y = input.height - 1;
        
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = (src_y * width + x) * channels + c;
                int dst_idx = (y * width + x) * channels + c;
                extended_input[dst_idx] = input.data[src_idx];
            }
        }
    }
    
    unsigned char *d_input, *d_output;
    int part_size = width * extended_height * channels;
    cudaMalloc(&d_input, part_size);
    cudaMalloc(&d_output, part_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(d_input, extended_input.data(), part_size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (extended_height + blockSize.y - 1) / blockSize.y);
    
    if (filter_type == "blur") {
        gaussian_blur_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, extended_height, channels);
    } else if (filter_type == "edge") {
        sobel_edge_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, extended_height, channels);
    } else if (filter_type == "denoise") {
        median_filter_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, extended_height, channels);
    }
    
    cudaDeviceSynchronize();
    cudaMemcpy(extended_output.data(), d_output, part_size, cudaMemcpyDeviceToHost);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float time = duration.count() / 1000.0f;
    
    // Копируем только основную часть (без границ) в выходное изображение
    for (int y = 1; y < extended_height - 1; y++) {
        int src_idx = (y * width) * channels;
        int dst_idx = ((start_row + y - 1) * width) * channels;
        memcpy(&output.data[dst_idx], &extended_output[src_idx], width * channels);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return time;
}

// Функция-обертка для запуска в отдельном потоке
std::pair<float, bool> process_gpu_wrapper(const Image& input, Image& output, const std::string& filter_type, 
                                          int gpu_id, int start_row, int end_row) {
    try {
        float time = process_part_parallel(input, output, filter_type, gpu_id, start_row, end_row);
        return std::make_pair(time, true);
    } catch (const std::exception& e) {
        std::cerr << "GPU " << gpu_id << " error: " << e.what() << std::endl;
        return std::make_pair(0.0f, false);
    }
}

// Основная функция с ПАРАЛЛЕЛЬНЫМ выполнением
void apply_filter_parallel_gpu(const Image& input, Image& output, const std::string& filter_type, 
                              float& total_time, float& time_gpu0, float& time_gpu1) {
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data.resize(input.width * input.height * input.channels);
    
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    
    if (num_gpus < 2) {
        std::cerr << "ERROR: Need at least 2 GPUs for multi-GPU processing!" << std::endl;
        return;
    }
    
    // БАЛАНСИРОВКА НАГРУЗКИ 50%/50% для параллельного выполнения
    int mid_row = input.height / 2;
    
    // GPU 0: верхняя половина
    int start1 = 0;
    int end1 = mid_row;
    
    // GPU 1: нижняя половина
    int start2 = mid_row;
    int end2 = input.height;
    
    std::cout << "Parallel execution: GPU0=" << (end1-start1) << " rows (" 
              << (end1-start1)*100/input.height << "%), "
              << "GPU1=" << (end2-start2) << " rows (" 
              << (end2-start2)*100/input.height << "%)" << std::endl;
    
    // ЗАПУСК В ОТДЕЛЬНЫХ ПОТОКАХ - ПАРАЛЛЕЛЬНО!
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::future<std::pair<float, bool>> future0 = std::async(std::launch::async, 
        process_gpu_wrapper, std::ref(input), std::ref(output), filter_type, 0, start1, end1);
    
    std::future<std::pair<float, bool>> future1 = std::async(std::launch::async, 
        process_gpu_wrapper, std::ref(input), std::ref(output), filter_type, 1, start2, end2);
    
    // Ждем завершения обоих потоков
    auto result0 = future0.get();
    auto result1 = future1.get();
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    total_time = duration.count() / 1000.0f;
    
    if (result0.second) time_gpu0 = result0.first;
    else time_gpu0 = 0;
    
    if (result1.second) time_gpu1 = result1.first;
    else time_gpu1 = 0;
}

bool write_png(const std::string& filename, const Image& image) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) return false;
    
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return false; }
    
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, NULL); fclose(fp); return false; }
    
    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(fp); return false; }
    
    png_init_io(png, fp);
    int color_type = PNG_COLOR_TYPE_RGB;
    png_set_IHDR(png, info, image.width, image.height, 8, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    
    std::vector<png_bytep> row_pointers(image.height);
    for (int y = 0; y < image.height; y++) {
        row_pointers[y] = const_cast<png_bytep>(&image.data[y * image.width * image.channels]);
    }
    
    png_write_image(png, row_pointers.data());
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return true;
}

bool read_png(const std::string& filename, Image& image) {
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) return false;
    
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return false; }
    
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return false; }
    
    if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, NULL); fclose(fp); return false; }
    
    png_init_io(png, fp);
    png_read_info(png, info);
    
    image.width = png_get_image_width(png, info);
    image.height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);
    
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    
    png_read_update_info(png, info);
    
    image.channels = 3;
    image.data.resize(image.width * image.height * image.channels);
    
    std::vector<png_bytep> row_pointers(image.height);
    for (int y = 0; y < image.height; y++) {
        row_pointers[y] = &image.data[y * image.width * image.channels];
    }
    
    png_read_image(png, row_pointers.data());
    png_read_end(png, NULL);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=== FINAL TECHNICAL ANALYSIS WITH NOISE ===" << std::endl;
    
    // ВЫВОД ТЕХНИЧЕСКИХ ХАРАКТЕРИСТИК ПЕРВЫМ ДЕЛОМ
    print_gpu_specs();
    
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    std::cout << "\nAvailable GPUs: " << num_gpus << std::endl;
    
    std::string input_file;
    Image input_image;
    
    if (argc > 1) {
        input_file = argv[1];
        std::cout << "Using provided image: " << input_file << std::endl;
        if (!read_png(input_file, input_image)) {
            std::cerr << "ERROR: Cannot read input image!" << std::endl;
            return 1;
        }
    } else {
        input_file = "technical_input_with_noise.png";
        std::cout << "No image provided, creating test image WITH NOISE" << std::endl;
        input_image = Image(610, 381, 3);
        create_test_image_with_noise(input_image);
        write_png(input_file, input_image);
    }
    
    std::cout << "Image loaded: " << input_image.width << "x" << input_image.height
              << " (" << input_image.channels << " channels)" << std::endl;
    std::cout << "Total pixels: " << input_image.width * input_image.height << std::endl;
    std::cout << "Data size: " << (input_image.width * input_image.height * input_image.channels) / 1024 << " KB" << std::endl;
    
    std::vector<std::string> filters = {"blur", "edge", "denoise"};
    
    std::cout << "\n=== PARALLEL PROCESSING RESULTS ===" << std::endl;
    
    for (const auto& filter : filters) {
        std::cout << "\n--- " << filter << " FILTER ---" << std::endl;
        
        Image output;
        float total_time, time_gpu0, time_gpu1;
        
        apply_filter_parallel_gpu(input_image, output, filter, total_time, time_gpu0, time_gpu1);
        
        // РАСЧЕТЫ И АНАЛИЗ
        float max_gpu_time = std::max(time_gpu0, time_gpu1);
        float min_gpu_time = std::min(time_gpu0, time_gpu1);
        float speedup = max_gpu_time / total_time;
        float performance_ratio = time_gpu0 / time_gpu1;
        
        std::cout << "GPU 0 Time: " << time_gpu0 << " ms" << std::endl;
        std::cout << "GPU 1 Time: " << time_gpu1 << " ms" << std::endl;
        std::cout << "Performance Ratio (GPU0/GPU1): " << performance_ratio << "x" << std::endl;
        std::cout << "Total Parallel Time: " << total_time << " ms" << std::endl;
        std::cout << "Theoretical Min Time: " << max_gpu_time << " ms" << std::endl;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
        std::cout << "Efficiency: " << (speedup / 2.0 * 100.0) << "%" << std::endl;
        
        // АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ
        if (performance_ratio > 1.5) {
            std::cout << "ANALYSIS: GPU 0 is " << performance_ratio << "x slower than GPU 1" << std::endl;
            std::cout << "POSSIBLE REASONS: Different PCIe slots, background processes, thermal throttling" << std::endl;
        } else if (performance_ratio < 0.67) {
            std::cout << "ANALYSIS: GPU 1 is " << (1.0/performance_ratio) << "x slower than GPU 0" << std::endl;
        } else {
            std::cout << "ANALYSIS: GPUs have similar performance" << std::endl;
        }
        
        std::string output_file = "technical_with_noise_" + filter + ".png";
        if (write_png(output_file, output)) {
            std::cout << "Saved: " << output_file << std::endl;
        }
    }
    
    std::cout << "\n=== TECHNICAL ANALYSIS WITH NOISE COMPLETE ===" << std::endl;
    return 0;
}
EOF

echo "=== Compiling version with noise ==="
nvcc -arch=sm_70 -std=c++11 -o final_technical_with_noise final_technical_with_noise.cu -lpng

echo "=== Running with generated image (with noise) ==="
./final_technical_with_noise

echo "=== Checking results ==="
ls -la technical_with_noise_*.png
file technical_with_noise_*.png
ls -lh technical_with_noise_*.png

echo "=== Job completed ==="
