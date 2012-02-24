#include "cutil.hpp"
#include <vector>
#include <math.h>


typedef std::vector<double> doubleary;

typedef struct {
  doubleary *rary;
  doubleary *gary;
  doubleary *bary;
  double rsum;
  double gsum;
  double bsum;
} std_t;

double calc_luminance(unsigned char r,unsigned b, unsigned g);
double triangular_function(double luminance);
std_t* calc_standard_deviation(rgbimage_t* img,int subwidth, int subheight, double subblocksize);
double* calc_luminance_weight(rgbimage_t* img,std_t* stdev,int subwidth, int subheight, double subblocksize);
std_t* create_std();
void delete_std(std_t* p);

double* calc_sdlwgw(rgbimage_t* img, int subwidth, int subheight){
  double* gains = new double[3]();
  double subblocksize = subwidth*subheight;

  std_t* stdev = calc_standard_deviation(img,subwidth,subheight,subblocksize);
  double* sdlwa = calc_luminance_weight(img, stdev, subwidth, subheight, subblocksize);
  delete_std(stdev);
  double sdlwa_avg = (sdlwa[0]+sdlwa[1]+sdlwa[2])/3.0;
  gains[0] = sdlwa_avg/sdlwa[0];
  gains[1] = sdlwa_avg/sdlwa[1];
  gains[2] = sdlwa_avg/sdlwa[2];
  delete[] sdlwa;
  return gains;
}

std_t* calc_standard_deviation(rgbimage_t* img,int subwidth, int subheight, double subblocksize){
  std_t* stdev = create_std();
  //calurete standard deviation for each subblocks
  for(int i=0;i<img->height;i+=subheight){
    for(int j=0;j<img->width;j+=subwidth){
      double avg_r=0;
      double avg_g=0;
      double avg_b=0;
      //iterate pixels in subblock
      for(int k=i;k<subheight;++k){
        for(int l=j;l<subwidth;++l){
          avg_r+=img->r[k*img->width+l];
          avg_g+=img->g[k*img->width+l];
          avg_b+=img->b[k*img->width+l];
        }
      }
      avg_r = avg_r/subblocksize;
      avg_g = avg_g/subblocksize;
      avg_b = avg_b/subblocksize;
      double std_r=0;
      double std_g=0;
      double std_b=0;
      //iterate pixels in subblock
      for(int k=i;k<subheight;++k){
        for(int l=j;l<subwidth;++l){
          std_r+=pow(img->r[k*img->width+l]-avg_r,2.0);
          std_g+=pow(img->g[k*img->width+l]-avg_g,2.0);
          std_b+=pow(img->b[k*img->width+l]-avg_b,2.0);
        }
      }
      std_r = sqrt(std_r/(subblocksize-1.0));
      std_g = sqrt(std_g/(subblocksize-1.0));
      std_b = sqrt(std_b/(subblocksize-1.0));
      stdev->rary->push_back(std_r);
      stdev->gary->push_back(std_g);
      stdev->bary->push_back(std_b);
      stdev->rsum+=std_r;
      stdev->gsum+=std_g;
      stdev->bsum+=std_b;
    }
  }
  return stdev;
}

double* calc_luminance_weight(rgbimage_t* img,std_t* stdev,int subwidth, int subheight, double subblocksize){
  double* sdlwa = new double[3]();
  int bid = 0;
  for(int i=0;i<img->height;i+=subheight){
    for(int j=0;j<img->width;j+=subwidth){
      doubleary wlumi_Ary;
      double wlumi_sum=0.0;
      //iterate pixels in subblock
      for(int k=i;k<subheight;++k){
        for(int l=j;l<subwidth;++l){
          unsigned char r=img->r[k*img->width+l];
          unsigned char g=img->g[k*img->width+l];
          unsigned char b=img->b[k*img->width+l];
          double luminance = calc_luminance(r,g,b);
          double w_luminance = triangular_function(luminance);
          wlumi_sum+=w_luminance;
          wlumi_Ary.push_back(w_luminance);
        }
      }
      int subpid=0;
      double lumi_r = 0;
      double lumi_g = 0;
      double lumi_b = 0;
      //iterate pixels in subblock
      for(int k=i;k<subheight;++k){
        for(int l=j;l<subwidth;++l){
          lumi_r += (wlumi_Ary[subpid]/wlumi_sum)*img->r[k*img->width+l];
          lumi_g += (wlumi_Ary[subpid]/wlumi_sum)*img->g[k*img->width+l];
          lumi_b += (wlumi_Ary[subpid]/wlumi_sum)*img->b[k*img->width+l];
          ++subpid;//increment pixel id
        }
      }
      sdlwa[0] += (stdev->rary->at(bid)/stdev->rsum)*lumi_r;
      sdlwa[1] += (stdev->gary->at(bid)/stdev->gsum)*lumi_g;
      sdlwa[2] += (stdev->bary->at(bid)/stdev->bsum)*lumi_b;
      ++bid;//increment block id
    }
  }
  return sdlwa;
}

double calc_luminance(unsigned char r,unsigned b, unsigned g){
  return (0.298912*r+0.586611*g+0.114478*b);
}

double triangular_function(double luminance){
  if(luminance<160){
    return luminance*(1.0/160.0);
  }else{
    return 1.0-(luminance-160)*(1.0/160.0);
  }
}

std_t* create_std(){
  std_t* stdev = new std_t;
  stdev->rary = new doubleary;
  stdev->gary = new doubleary;
  stdev->bary = new doubleary;
  stdev->rsum = 0.0;
  stdev->gsum = 0.0;
  stdev->bsum = 0.0;
  return stdev;
}

void delete_std(std_t* p){
  delete p->rary;
  delete p->gary;
  delete p->bary;
  delete p;
}

void delete_doubleptr(float* p){
  delete[] p;
}
