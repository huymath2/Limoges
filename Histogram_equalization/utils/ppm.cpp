#include "ppm.hpp"
#include <cstring>

PPMBitmap::PPMBitmap( const int width, const int height ) 
	: _ptr( NULL ), _width( width ), _height ( height )
{
	_ptr = new uchar[_width * _height * 3];
	std::memset( _ptr, 0, _width * _height * 3 );
}

PPMBitmap::PPMBitmap( const char *const name ) 
	: _ptr( NULL ), _width( 0 ), _height ( 0 )
{
	std::ifstream file( name, std::ios::in | std::ios::binary );
	if ( !file ) {
		std::cerr << "Cannot open PPM file " << name << std::endl;
		exit( EXIT_FAILURE );
	}
	std::string MagicNumber, line;
	
	// MagicNumber
	getLine( file, line );
	std::istringstream iss1( line );
	iss1 >> MagicNumber;
	if ( MagicNumber != "P6" ) { // Binary ? or nothing ?
		std::cerr << "Error reading PPM file " << name << ": unknown Magic Number \"" << MagicNumber 
			 << "\". Only P6 is supported" << std::endl << std::endl;
		exit( EXIT_FAILURE );
	}

	// Image size
	getLine( file, line );
	std::istringstream iss2( line );
	iss2 >> _width >> _height;
	if ( _width <= 0 || _height <= 0 ) {
		std::cerr << "Wrong image size " << _width << " x " << _height << std::endl;
		exit( EXIT_FAILURE );
	}

	// Max channel value
	int maxChannelVal;
	getLine( file, line );
	std::istringstream iss3( line );
	iss3 >> maxChannelVal;
	if ( maxChannelVal > 255 ) {
		std::cerr << "Max channel value too high in " << name << std::endl;
		exit( EXIT_FAILURE );
	}

	size_t size = _height * _width * 3;
	// Allocate pixels
	_ptr = new uchar[size];

	// Read pixels
	for ( int y = _height; y-- > 0; ) { // Reading each line
		file.read( reinterpret_cast<char *>( _ptr + y * _width * 3), _width * 3 );
	}
}

void PPMBitmap::getLine( std::ifstream &file, std::string &s ) const {
	for (;;) {
		if (!std::getline( file, s ) ) {
			std::cerr << "Error reading PPM file" << std::endl;
			exit( EXIT_FAILURE );
		}
		std::string::size_type index = s.find_first_not_of( "\n\r\t " );
		if ( index != std::string::npos && s[index] != '#' )
			break;
	}
}

void PPMBitmap::saveTo( const char *name ) const {
	std::ofstream file( name, std::ios::out | std::ios::trunc | std::ios::binary );
	file << "P6" << std::endl;						// Magic Number !
	file << _width << " " << _height << std::endl;// Image size
	file << "255" << std::endl;						// Max R G B

	uchar *ptr = _ptr;

	for ( int y = _height; y-- > 0; ) { // Writing each line
		file.write( (char *)( _ptr + y * _width * 3), _width * 3 ); 
	}
	file.close();
}

