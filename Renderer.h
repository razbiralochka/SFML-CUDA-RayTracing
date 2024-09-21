#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <windows.h>

void initRenderer(int w, int h);
void releaseRenderer();

void renderFrame(uint8_t* frame, float t);





__global__ void drawPixel(uint8_t* result, float t);

__device__  void traceRay(float* origin, float* direction, float* color, int geneneration, int source);

__device__ void floorHit(float* origin, float* direction, float* color, int geneneration);

__device__ void sphereHit(float* origin, float* direction, float* color, int geneneration, float oppacity);