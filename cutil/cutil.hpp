#ifndef COLORCORRECT_CUTIL_CUTIL_H_
#define COLORCORRECT_CUTIL_CUTIL_H_
#include <stdlib.h>

extern "C"{
  typedef struct {
    int width;
    int height;
    unsigned char *r;
    unsigned char *g;
    unsigned char *b;
  } rgbimage_t;

  /* Each calc_* returns a heap allocated double[3] = {gain_r, gain_g, gain_b}
     that must be released with delete_doubleptr(). */
  double* calc_sdwgw(rgbimage_t* img, int subwidth, int subheight);
  double* calc_sdlwgw(rgbimage_t* img, int subwidth, int subheight);
  double* calc_lwgw(rgbimage_t* img, int subwidth, int subheight);
  /* Monte-Carlo ACE: samples random pixels (shared by every target pixel).
     Overwrites img in place. seed selects the random subset; distances
     shorter than min_distance are clamped to it so that a sample right next
     to a pixel cannot dominate its result. */
  void calc_ace(rgbimage_t* img, int samples, double slope, double limit,
                unsigned int seed, double min_distance);
  void delete_doubleptr(double* p);
  /* Random Spray Retinex: for every pixel `sprays` random sprays of `points`
     pixels with radial density 1/r (radius `radius`) are drawn; the output
     is the mean over sprays of I_c(x) / max of I_c over the spray, scaled to
     [0, 255].  `in` is read, `out` written (same size).  threads <= 0 uses
     the hardware concurrency. */
  void calc_rsr(const rgbimage_t* in, rgbimage_t* out, int sprays, int points,
                double radius, unsigned int seed, int threads);
}

#endif
