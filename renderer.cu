#include "Renderer.h"

uint8_t* readyFrame;

int frame_size;
int N;
int width;
int height;


__device__ float lightSourse[3] = { -3 ,2, 5 };
__device__ float sphere[4] = { 0 ,0, 5, 1 };
__device__ float scene_time;


void initRenderer(int w, int h)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout <<"Device name: " << props.name << "\n";
    cudaSetDevice(0);
    frame_size = w * h * 4 * sizeof(uint8_t);
    N = w * h * 4;
    height = h;
    width = w;
    cudaMalloc(&readyFrame, frame_size);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << cudaGetErrorName(error) << std::endl;
    }
}

void releaseRenderer()
{   
    cudaFree(&readyFrame);
    std::cout << "release resources\n";

}


void renderFrame(uint8_t* frame, float t)
{   
    dim3 threads(20, 20);
    dim3 blocks(width/20, height/20);
    
    
    drawPixel <<<blocks, threads>>>(readyFrame, t);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << cudaGetErrorName(error) << std::endl;
    }
 
    cudaMemcpy(frame, readyFrame, frame_size, cudaMemcpyDeviceToHost);
    
}


__global__ void drawPixel(uint8_t* result, float t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    scene_time = t*5e-3;

    int id = 4 * (1000 * i + j);


    float origin[3] = {0,0,0};
    float pixelColor[3] = { 0 };

    float x = 0.002 * j - 1;
    float y = 1 - 0.002 * i;
    float z = 1;
    float direction[3] = { x,y,z };
   
    sphere[1] = 1 - cos(scene_time);

    traceRay(origin, direction, pixelColor, 1, 0);

    result[id]   =  255 * pixelColor[0];
    result[id+1] =  255 * pixelColor[1];
    result[id+2] =  255 * pixelColor[2];
    result[id+3] =  255;
}




__device__  void traceRay(float* origin, float* direction, float* color, int geneneration, int source)
{
    float l = sqrtf(powf(direction[0], 2) + powf(direction[1], 2) + powf(direction[2], 2));
    direction[0] /= l;
    direction[1] /= l;
    direction[2] /= l;
    
    

    if (source != 1)
        floorHit(origin, direction, color, geneneration);

    if(source != 2)
        sphereHit(origin, direction, color, geneneration, 0);
}


__device__ void floorHit(float* origin, float* direction, float* color, int geneneration)
{
  
  if(direction[1] < 0)
  { 
    float light = 0;
    float Normal[3] = { 0, 1, 0 };
    float rayHit[3];
    float brightness = 0;
    
    float toLight[3];
    float t = -1 / direction[1];
    float l2 = 0;
    for (size_t i = 0; i < 3; i++)
    {
        rayHit[i] = origin[i] + direction[i] * t;
        toLight[i] = lightSourse[i] - rayHit[i];
        brightness += Normal[i] * toLight[i];
        l2 += powf(toLight[i],2);
    }

    if (direction[1] < 0 && t > 0)
    {
        float vel = scene_time;
        //light = sinf(5 * rayHit[0] + 100 * sin(1e-4 * scene_time)) * cosf(5 * rayHit[2] + 100 * cos(1e-4 * scene_time));
        light = sinf(5 * rayHit[0] + vel) * cosf(5 * rayHit[2]);
        if (light > 0)
            light = 1;
        else
            light = 0;

        if (pow(rayHit[0], 2) + pow(rayHit[2] - 5, 2) > 10000)
            light = 0;

    }
   
    

    if (brightness/sqrt(l2) < 0)
    {
        brightness = 0;
    }
   
    float shadow[3] = { 0 };
    float colorsum = 0;
    if (direction[1] < 0 && t > 0 && geneneration < 3)
    {
        traceRay(rayHit, toLight, shadow, geneneration + 1, 1);
    }
   
    colorsum = powf(shadow[0], 2) + powf(shadow[1], 2) + powf(shadow[2], 2);
    

    
    for (size_t i = 0; i < 3; i++)
    {
        color[i] = light * (0.1 + 0.9 * brightness / sqrt(l2));
        
        if (colorsum > 0) 
            color[i] *= 0.1;
     
    }
   
  }
  else
  for (size_t i = 0; i < 3; i++)
  {
      color[i] = 0;

  }

}

__device__ void sphereHit(float* origin, float* direction, float* color, int geneneration, float oppacity)
{
    float a = 0;
    float b = 0;
    float c = 0;
    float d = 0;
    
    
    for (size_t i = 0; i < 3; i++)
    {
        a += powf(direction[i], 2);
        b += 2 * direction[i] * (origin[i] - sphere[i]);
        c += powf(origin[i] - sphere[i], 2);
    }

    c -= powf(sphere[3], 2);

    d = b * b - 4 * a * c;

    if (d >= 0)
    {

        float Normal[3];
        float rayHit[3];
        float brightness = 0;
        float toLight[3];
        float t;
        float l2 = 0;
        float l1 = 0;
        t = (-b - sqrtf(d)) / (2 * a);
       
        for (size_t i = 0; i < 3; i++)
        {
            rayHit[i] = origin[i] + direction[i] * t;
            Normal[i] = rayHit[i] - sphere[i];
            toLight[i] = lightSourse[i] - rayHit[i];
            l1 += Normal[i] * Normal[i];
            l2 += toLight[i] * toLight[i];
            
        }
      
        float dotProd = 0;
        float glare[3];
        float reflections[3];
        
        for (size_t i = 0; i < 3; i++)
            dotProd += (Normal[i] * toLight[i]);
    
        l1 = 0;
        for (size_t i = 0; i < 3; i++)
        {
            glare[i] = toLight[i] - 2*dotProd*Normal[i];
            l1 += powf(glare[i], 2);
        }

        

        dotProd = 0;
        for (size_t i = 0; i < 3; i++)
            brightness += (-direction[i] * glare[i]) / sqrt(l1);
        
       
        
        if (brightness < 0)
        {
            brightness = 0;
        }
        
        color[0] = 1;
        color[1] = 0;
        color[2] = 0;
        
        
        dotProd = 0;
        for (size_t i = 0; i < 3; i++)
            dotProd += (Normal[i] * direction[i]);
        
        for (size_t i = 0; i < 3; i++)
        {
            reflections[i] = -(2 * dotProd * Normal[i] - direction[i]);
            
        }

        

        float refColors[3] = { 0 };

        if (geneneration < 3)
           traceRay(rayHit, reflections, refColors, geneneration + 1, 2);




        for (size_t i = 0; i < 3; i++)
        {
            color[i] = 0.2 * color[i] + 0.8 * pow(brightness,11);
            color[i] = 0.75 * color[i] + 0.25 * refColors[i];
        }


    }
   
        
    
}