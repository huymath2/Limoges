#include "utils/chronoCPU.hpp"
#include <iostream>
#include "image.hpp"
#include <iostream>
#include <cmath>

using namespace std;

#define L 256

static double Min(double a, double b) {
	return a <= b ? a : b;
}

static double Max(double a, double b) {
	return a >= b ? a : b;
}

void rgb2hsv_CPU(unsigned char *pixels, double *h_h, double *h_s, double *h_v, int imgSize)
{
	for(int i = 0; i < imgSize; ++i)
	{
		double r = (double) pixels[3 * i];
		double g = (double) pixels[3 * i + 1];
		double b = (double) pixels[3 * i + 2];

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
		//save to class HSV
		h_h[i] = h;
		h_s[i] = s;
		h_v[i] = v;
	}
}

void hsv2rgb_CPU(double *h_h, double *h_s, double *h_v, unsigned char *pixels, const int imgSize)
{

	for(int i = 0; i < imgSize; ++i)
	{
		double h = (double)h_h[i];
		double s = (double)h_s[i];
		double v = (double)h_v[i];
		if(h>360 || h<0 || s>100 || s<0 || v>100 || v<0){
			cout<<"The givem HSV values are not in valid range"<<endl;
			continue;
		}
		int H = round(h/60);
		double C = s*v;
		double X = C*(1 - abs((H/60)%2 - 1));
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
		pixels[3 * i] 		= (r+m)*255;
		pixels[3 * i + 1]  	= (g+m)*255;
		pixels[3 * i + 2]  	= (b+m)*255;
		// if(i < 10)
		// {
		// 	printf("%d: %lf %lf %lf\n", i, r, g, b);
		// 	printf("%d: %d %d %d\n", i, pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2]);
		// }
	}


}

void histogram_CPU(double *v, unsigned int *outHisto, const int imgSize)
{
	
	for(int i = 0; i < imgSize; ++i)
	{
		int V = (int)round(v[i] * (L - 1));
		outHisto[V]++;
		//if(inV[i] == 100) printf("%d: %d\n", i, inV[i]);
	}
}

void repart_CPU(unsigned int *histoV, unsigned int *r)
{
	for(int i = 0; i < L; ++i)
	{
		for(int j = 0; j <= i; ++j)
		{
			r[i] += histoV[j];
		}
		//printf("%d: %d\n", i, r[i]);
	}
}

void equalization_CPU(double *v, unsigned int *r, double *t, const int imgSize)
{
	for(int i = 0; i < imgSize; ++i)
	{
		int V = round(v[i] * (L - 1));
		t[i] = (double)(L - 1)/(L*imgSize) * r[V];
	}
}


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
    double *h, *s, *v, *t;
    h = (double*) malloc(imgSize * sizeof(double));
    s = (double*) malloc(imgSize * sizeof(double));
    v = (double*) malloc(imgSize * sizeof(double));
    unsigned int *hist, *r;
    hist = (unsigned int*) malloc(L * sizeof(unsigned int));
    r = (unsigned int*) malloc(L * sizeof(unsigned int));
	t = (double*) malloc(imgSize * sizeof(double));

	memset(hist, 0, L*sizeof(unsigned int));
	memset(r, 0, L*sizeof(unsigned int));



	ChronoCPU chr;
	chr.start();
	rgb2hsv_CPU(pixels, h, s, v, imgSize);
	histogram_CPU(v, hist, imgSize);
	repart_CPU(hist, r);
	equalization_CPU(v, r, t, imgSize);
	hsv2rgb_CPU(h, s, t, output_Img._pixels, imgSize);

	chr.stop();
	// tab_printf(hist);
	// cout << endl << endl;
	// tab_printf(r);
	// for(int i = imgSize; i > imgSize - 10; --i)
	// {
	// 	printf("%d: %d\n", i, output_Img._pixels[3*i]);
	// }

	output_Img.save("output_CPU.png");
	std::cout << "-> Done : " << chr.elapsedTime() << " ms" << std::endl << std::endl;

}