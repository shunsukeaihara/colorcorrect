#include <stdlib.h>

extern "C"{
  typedef struct {
    int width;
    int height;
    unsigned char *r;
    unsigned char *g;
    unsigned char *b;
  } rgbimage_t;

  double* calc_sdlwgw(rgbimage_t* img, int subwidth, int subheight);
  void delete_doubleptr(double* p);
}
