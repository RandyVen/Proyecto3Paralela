
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "pgm.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define M_PI 3.141592653589793238462643383279502884L

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

#define CUDA_CHECK_RETURN(value)                                          \
    {                                                                     \
        cudaError_t _m_cudaStat = value;                                  \
        if (_m_cudaStat != cudaSuccess)                                   \
        {                                                                 \
            fprintf(stderr, "Error %s at line %d in file %s\n",           \
                    cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
            exit(1);                                                      \
        }                                                                 \
    }

void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++)
        {
            int idx = j * w + i;
            if (pic[idx] > 0)
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++)
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
}

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{

    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID > w * h)
        return;
    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    __syncthreads();

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {

            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;

            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

int main(int argc, char **argv)
{
    int i;

    PGMImage inImg(argv[1]);
    cudaEvent_t start, stop;
    float time;

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++)
    {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels;

    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = ceil(w * h / 256);

    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

    GPU_HoughTranConst<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

    cudaDeviceSynchronize();

    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
    // Calcular el umbral para dibujar las líneas
    int threshold = 5000;

    // Calcular el promedio de los pesos
    float sum = 0;
    for (i = 0; i < degreeBins * rBins; i++)
    {
        sum += h_hough[i];
    }
    float average = sum / (degreeBins * rBins);

    // Calcular la desviación estándar
    float variance = 0;
    for (i = 0; i < degreeBins * rBins; i++)
    {
        variance += pow(h_hough[i] - average, 2);
    }
    variance /= (degreeBins * rBins);
    float stdDev = sqrt(variance);

    // Crear una copia de la imagen de entrada para dibujar las líneas
    unsigned char *outputImage = new unsigned char[w * h * 3]; // 3 canales: RGB
    memcpy(outputImage, h_in, sizeof(unsigned char) * w * h);

    // Dibujar las líneas cuyo peso es mayor que el umbral
    for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            int weight = h_hough[rIdx * degreeBins + tIdx];
            if (weight > threshold)
            {
                float r = rIdx * rScale - rMax;
                float theta = tIdx * radInc;

                for (int x = 0; x < w; x++)
                {
                    int y = static_cast<int>((r - x * cos(theta)) / sin(theta));
                    if (y >= 0 && y < h)
                    {
                        int idx = (y * w + x) * 3; // Índice para los 3 canales RGB
                        outputImage[idx] = 255;    // R: dibujar la línea en rojo
                        outputImage[idx + 1] = 0;  // G: dejar en 0
                        outputImage[idx + 2] = 0;  // B: dejar en 0
                    }
                }
            }
        }
    }

    // Guardar la imagen resultante en formato JPG
    stbi_write_jpg("output.jpg", w, h, 3, outputImage, w * 3); // Ajustar el tamaño de la imagen

    // Liberar la memoria utilizada
    delete[] outputImage;

    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (cpuht[i] != h_hough[i])
            printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
    printf("Done!\n");
    printf("EXEC TIME:  %.10f s \n", time / 1000.0);

    cudaFree((void *)d_Cos);
    cudaFree((void *)d_Sin);
    cudaFree((void *)d_in);
    cudaFree((void *)d_hough);
    free(h_hough);
    free(cpuht);
    free(pcCos);
    free(pcSin);
    cudaDeviceReset();

    return 0;
}