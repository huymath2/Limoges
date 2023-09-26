#ifndef __PP_HPP__
#define __PP_HPP__

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

typedef unsigned char uchar;

struct RGBcol {
	uchar _r, _g, _b;

	explicit RGBcol() : _r( 0 ), _g ( 0 ), _b ( 0 ) {}
	explicit RGBcol( const uchar red, const uchar green, const uchar blue ) 
		: _r( red ), _g( green ), _b( blue ) {};
};

class PPMBitmap {
public:
	PPMBitmap() = delete; // undefined
	PPMBitmap( const int width, const int height );
	PPMBitmap( const char *const name );
	~PPMBitmap() { delete _ptr; }

	int		getWidth()	const	{ return _width; }
	int		getHeight()	const	{ return _height; }
	size_t	getSize()	const	{ return _width * _height * 3; }
	uchar *	getPtr()	const	{ return _ptr; }

	const RGBcol &getPixel( const unsigned x, const unsigned y ) const {
		return _pixels[ x + y * _width];
	}
	void setPixel( const unsigned x, const unsigned y, const RGBcol &RGBcol ) {
		_pixels[ x + y * _width ] = RGBcol;
	}
	
	void getLine( std::ifstream &file, std::string &s ) const;

	void saveTo( const char *name ) const;

private:
	union {
		uchar	*_ptr;
		RGBcol	*_pixels;
	};

	int _width;
	int _height;
};

#endif // __PP_HPP__

