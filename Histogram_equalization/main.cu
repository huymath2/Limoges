#include <iostream>
#include <string>
#include <iomanip>

#include "image.hpp"
#include "utils/chronoGPU.hpp"
#include "utils/commonCUDA.hpp"

#define L 256 //max gray scale
#define NUMBER_THREAD 512
#define RESOLUTION 0.5

// ------------- Device function -----------------//
__device__ double Min(double a, double b) {
	return a <= b ? a : b;
}

__device__ double Max(double a, double b) {
	return a >= b ? a : b;
}

// ---------------- Kernel Function -----------------   //


__global__ void kernelCompute_rgb2hsv( const unsigned char *dev_pixels_in, double *dev_h, double *dev_s, double *dev_v, const int imgSize ) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	double r, g, b;
    if(tid < imgSize)
    {
        r = (double)dev_pixels_in[3*tid];
		g = (double)dev_pixels_in[3*tid+1];
		b = (double)dev_pixels_in[3*tid+2];

        // R, G, B values are divided by 255
		// to change the range from 0..255 to 0..1
		r = r / 255.0;
		g = g / 255.0;
		b = b / 255.0;
	
		double delta, min;
		double h = 0, s, v;

        min = Min(Min(r, g), b);
		v = Max(Max(r, g), b);
		delta = v - min;

		if (v == 0.0)
			s = 0;
		else
			s = delta / v;

		if (s == 0)
			h = 0.0;

		else
		{
			if (r == v)
				h = (g - b) / delta;
			else if (g == v)
				h = 2 + (b - r) / delta;
			else if (b == v)
				h = 4 + (r - g) / delta;

			h *= 60;

			if (h < 0.0)
				h = h + 360;
		}

        dev_h[tid] = h;
        dev_s[tid] = s;
        dev_v[tid] = v;
    } 
}

__global__ void kernelCompute_rgb2hsv_v2( const unsigned char *dev_pixels_in, double *dev_h, double *dev_s, double *dev_v, const int imgSize ) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	double r, g, b;
    while(tid < imgSize)
    {
        r = (double)dev_pixels_in[3*tid];
		g = (double)dev_pixels_in[3*tid+1];
		b = (double)dev_pixels_in[3*tid+2];

        // R, G, B values are divided by 255
		// to change the range from 0..255 to 0..1
		r = r / 255.0;
		g = g / 255.0;
		b = b / 255.0;
	
		double delta, min;
		double h = 0, s, v;

        min = Min(Min(r, g), b);
		v = Max(Max(r, g), b);
		delta = v - min;

		if (v == 0.0)
			s = 0;
		else
			s = delta / v;

		if (s == 0)
			h = 0.0;

		else
		{
			if (r == v)
				h = (g - b) / delta;
			else if (g == v)
				h = 2 + (b - r) / delta;
			else if (b == v)
				h = 4 + (r - g) / delta;

			h *= 60;

			if (h < 0.0)
				h = h + 360;
		}

        dev_h[tid] = h;
        dev_s[tid] = s;
        dev_v[tid] = v;

        tid += blockDim.x * gridDim.x;
    } 
}

__global__ void  kernelCompute_hsv2rgb(double *dev_h, double *dev_s, double *dev_tx, unsigned char *dev_pixels_out, const int imgSize )
{
    /// TODO
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < imgSize)
	{
		double h = dev_h[tid];
		double s = dev_s[tid];
		double v = dev_tx[tid];
		if(h>360 || h<0 || s>100 || s<0 || v>100 || v<0){
			printf("The givem HSV values are not in valid range");
			return;
		}
        //if(tid < 10) printf("%d: %lf\n", tid, (double)h/360);
        int H = round(h/60);
		double C = s*v;
		double X = C*(1-abs(H%2 - 1));
		double m = v - C;
		double r, g, b;
		
		if(h >= 0 && h < 60){
			r = C,g = X,b = 0;
		}
		else if(h >= 60 && h < 120){
			r = X,g = C,b = 0;
		}
		else if(h >= 120 && h < 180){
			r = 0,g = C,b = X;
		}
		else if(h >= 180 && h < 240){
			r = 0,g = X,b = C;
		}
		else if(h >= 240 && h < 300){
			r = X,g = 0,b = C;
		}
		else{
			r = C,g = 0,b = X;
		}
		dev_pixels_out[3 * tid] 		= (unsigned char)((r+m)*255);
		dev_pixels_out[3 * tid + 1]  	= (unsigned char)((g+m)*255);
		dev_pixels_out[3 * tid + 2]  	= (unsigned char)((b+m)*255);
		// if(tid < 10)
		// 	printf("%d: %f %f %f\n", tid, (r+m)*255, (g+m)*255, (b+m)*255);
	}
}



__global__ void kernelCompute_histogram( double *dev_v, unsigned int *dev_hist, const int imgSize )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < imgSize)
    {
        int V = (int)round(dev_v[tid] * (L - 1));
        __syncthreads();
		atomicAdd(&dev_hist[V], 1);
    } 
}

__global__ void kernelCompute_histogram_v2( double *dev_v, unsigned int *dev_hist, const int imgSize )
{
    /// TODO
	__shared__ int temp_cnt[L]; //chi chay trong 1 block
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int  i = threadIdx.x; i < L; i += blockDim.x){ 
		temp_cnt[i] = 0;
	}
	__syncthreads();
	if(tid < imgSize)
	{
        int V = (int)round(dev_v[tid] * (L - 1));
		atomicAdd(&temp_cnt[V], 1);
		
        __syncthreads();

		for (int j = threadIdx.x; j < L; j += blockDim.x) {
            atomicAdd(&dev_hist[j], temp_cnt[j]);
        }
	}
}


__global__ void kernelCompute_repart( unsigned int *dev_hist,  unsigned int *dev_r )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < L)
    {
        int sum = 0;
        for(int i = 0; i <= tid; ++i){
            sum += dev_hist[i];
        }
        dev_r[tid] = sum;
    } 
}

__global__ void kernelCompute_repart_v2( unsigned int *dev_hist,  unsigned int *dev_r )
{
    extern __shared__ unsigned int hist[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if(tid < L)
    {
        hist[tid] = dev_hist[tid];
        __syncthreads();
        int sum = 0;
        for(int i = 0; i <= tid; ++i){
            sum += hist[i];
        }
        dev_r[tid] = sum;
    } 
}


__global__ void kernelCompute_equalization( double *dev_v, unsigned int *dev_r, double *dev_tx, const int imgSize )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int V = (int)round(dev_v[tid] * (L - 1));
    if(tid < imgSize)
    {
        dev_tx[tid] = (double)(L - 1)/(L*imgSize) * dev_r[V];
        // if(tid < 10)
        // {
        //     printf("%d: %lf - %d\n", tid, dev_tx[tid], dev_r[V]);
        // }
    }
}









///////////////////////////////////////////////////////////


//  Support function    //
void printUsage() 
{
	std::cerr	<< "Usage: " << std::endl
			<< " \t -f <F>: <F> image file name" 			
		    << std::endl << std::endl;
	exit( EXIT_FAILURE );
}

//  Check value of table    //
void tab_printf(unsigned int *table)
{
    for(int i = 0; i < L; ++i)
    {
        printf("|%d: %d | ", i, table[i]);
    }
}

int main( int argc, char **argv ) 
{	
	char fileName[2048];

	// Parse program arguments
	if ( argc == 1 ) 
	{
		sscanf( "Chateau.png", "%s", &fileName);
	}

	for ( int i = 1; i < argc; ++i ) 
	{
		if ( !strcmp( argv[i], "-f" ) ) 
		{
			if ( sscanf( argv[++i], "%s", &fileName ) != 1 )
				printUsage();
		}
		else
			sscanf( "Chateau.png", "%s", &fileName);
	}

    // Reading image
    std::cout << "Loading image: " << fileName << std::endl;
	Image input_Img;
	input_Img.load( fileName );
	const int width		    = input_Img._width;
	const int height	    = input_Img._height;
	const int imgSize	    = width * height;
	const int channels 	    = input_Img._nbChannels;
    unsigned char *pixels   = input_Img._pixels;  

    std::cout << "Image has " << width << " x " << height << " pixels" << std::endl;
    // Intialize output image
    Image output_Img(width, height, channels);

    //--------- Intialize variable -----------//
    
    // test var //
    // double *h, *s, *v;
    // h = (double*) malloc(imgSize * sizeof(double));
    // s = (double*) malloc(imgSize * sizeof(double));
    // v = (double*) malloc(imgSize * sizeof(double));
    unsigned int *hist, *r;
    hist = (unsigned int*) malloc(imgSize * sizeof(unsigned int));
    r = (unsigned int*) malloc(imgSize * sizeof(unsigned int));

    // const var //
    int nb_RGB = width * height * channels;

    //  Device variable //
    unsigned char *dev_pixels_in, *dev_pixels_out;  //for input and output rgb 
    double *dev_h, *dev_s, *dev_v;                  //for saving hsv
    unsigned int *dev_hist, *dev_r;                 //for saving histogram and repart 
    double *dev_tx;                                 //for equalization

    //std::cout << "I am here" << std::endl;

    //------- Allocate memory on Device ------//
    HANDLE_ERROR(cudaMalloc(&dev_pixels_in, nb_RGB*sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc(&dev_pixels_out, nb_RGB*sizeof(unsigned char)));
    HANDLE_ERROR(cudaMalloc(&dev_h, imgSize*sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&dev_s, imgSize*sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&dev_v, imgSize*sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&dev_hist, L*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc(&dev_r, L*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc(&dev_tx, imgSize*sizeof(double)));

    //--------Intialize Value----------//
    cudaMemset(dev_hist, 0, L*sizeof(unsigned int));
    cudaMemset(dev_r, 0, L*sizeof(unsigned int));

    //------Copy from Host to Device---------//
    cudaMemcpy(dev_pixels_in, pixels, nb_RGB*sizeof(unsigned char), cudaMemcpyHostToDevice);



    //-------- Configure kernel -----------//
    ChronoGPU chr;
	chr.start();


    //-------- Lauch kernel ----------//
    
    //  Work 1: rgb2hsv
	kernelCompute_rgb2hsv<<< (imgSize + NUMBER_THREAD - 1)/NUMBER_THREAD, NUMBER_THREAD >>>(dev_pixels_in, dev_h, dev_s, dev_v, imgSize);
    //kernelCompute_rgb2hsv_v2<<< (imgSize + NUMBER_THREAD - 1)/NUMBER_THREAD, NUMBER_THREAD >>>(dev_pixels_in, dev_h, dev_s, dev_v, imgSize);
    //  Work 3: Histogram V
    //kernelCompute_histogram<<< (imgSize + NUMBER_THREAD - 1)/NUMBER_THREAD, NUMBER_THREAD >>>( dev_v, dev_hist, imgSize);
    kernelCompute_histogram_v2<<< (imgSize + NUMBER_THREAD - 1)/NUMBER_THREAD, NUMBER_THREAD >>>( dev_v, dev_hist, imgSize);
    //  Work 4: compute r
    //kernelCompute_repart<<< (L + NUMBER_THREAD - 1)/NUMBER_THREAD, NUMBER_THREAD >>> (dev_hist, dev_r);
    kernelCompute_repart_v2<<< (L + NUMBER_THREAD - 1)/NUMBER_THREAD, NUMBER_THREAD, L * sizeof(unsigned int) >>> (dev_hist, dev_r);

    //  Work 5: equalization
    kernelCompute_equalization <<< (imgSize + NUMBER_THREAD - 1)/NUMBER_THREAD, NUMBER_THREAD >>> (dev_v, dev_r, dev_tx, imgSize);

    //  Work 2: hsv2rgb
    kernelCompute_hsv2rgb <<< (imgSize + NUMBER_THREAD - 1)/NUMBER_THREAD, NUMBER_THREAD >>> (dev_h, dev_s, dev_tx, dev_pixels_out, imgSize);




    chr.stop();

    //------Copy from Device to Host--------//
    //HANDLE_ERROR( cudaMemcpy( h, dev_h, imgSize*sizeof(double), cudaMemcpyDeviceToHost ) );
    //HANDLE_ERROR( cudaMemcpy( s, dev_s, imgSize*sizeof(double), cudaMemcpyDeviceToHost ) );
    //HANDLE_ERROR( cudaMemcpy( v, dev_v, imgSize*sizeof(double), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( hist, dev_hist, L*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( r, dev_r, L*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( output_Img._pixels, dev_pixels_out, nb_RGB*sizeof(unsigned char), cudaMemcpyDeviceToHost ) );

    //---------test------------//
    // for(int i = 0; i < 10; ++i)
    // {
    //     printf("%d: %lf %lf %lf", i, h[i], s[i], v[i]);
    // }
    //tab_printf(hist);
    // std::cout << "I am here\n\n\n" << std::endl;
    // tab_printf(r);
    
    //---------- save to Output image--------//
    // for (int i = imgSize - 1; i > imgSize - 10; --i)
    // {
    //     printf("%d: %d %d %d\n", i, output_Img._pixels[i], output_Img._pixels[i + 1], output_Img._pixels[i + 2]);
    // }
    output_Img.save("output.png");

    std::cout << "-> Done : " << chr.elapsedTime() << " ms" << std::endl << std::endl;

    ///---- Free memory on Device -------//
	cudaFree(dev_pixels_in);
	cudaFree(dev_pixels_out);
    cudaFree(dev_h);
    cudaFree(dev_s);
    cudaFree(dev_v);
    cudaFree(dev_hist);
    cudaFree(dev_r);
    cudaFree(dev_tx);
}