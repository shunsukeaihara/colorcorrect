#include "cutil.hpp"
#include <vector>
#include <math.h>
#include <cfloat>
#include <random>
#include <algorithm>
#include <thread>

/*
 * Block based grey world variants from
 *   H.K. Lam, O.C. Au, C.W. Wong, "Automatic white balancing using luminance
 *   component and standard deviation of RGB components", ICASSP 2004.
 *
 * The image is split into sub-blocks of subwidth x subheight pixels.  Blocks
 * on the right/bottom border may be smaller than requested; they are kept and
 * weighted with their real pixel count instead of reading past the buffer.
 *
 * For every block i and channel c the block average mu_c(i) and the standard
 * deviation sigma_c(i) are computed.  The standard deviation weighted average
 *   SDWA_c = sum_i sigma_c(i) / sum_j sigma_c(j) * mu_c(i)
 * replaces the plain image mean of the grey world algorithm, and the gains are
 *   k_c = (SDWA_r + SDWA_g + SDWA_b) / 3 / SDWA_c.
 * In the luminance weighted variants mu_c(i) is a luminance weighted mean of
 * the block, using a triangular weighting function of the luminance Y.
 */

namespace {

struct BlockStats {
  std::vector<double> mean[3];  // block (weighted) means per channel
  std::vector<double> std[3];   // block standard deviations per channel
  double std_sum[3];
  BlockStats() { std_sum[0] = std_sum[1] = std_sum[2] = 0.0; }
};

const double kLuminancePeak = 160.0;

inline double luminance(unsigned char r, unsigned char g, unsigned char b) {
  // ITU-R BT.601 luma
  return 0.298912 * r + 0.586611 * g + 0.114478 * b;
}

// Triangular weighting function of the luminance with its peak at Y = 160
// (Lam et al. 2004): zero weight for black, maximum at 160 and decreasing
// again towards saturated (white) pixels.
inline double triangular_weight(double y) {
  if (y <= 0.0 || y >= 255.0) return 0.0;
  if (y < kLuminancePeak) return y / kLuminancePeak;
  return (255.0 - y) / (255.0 - kLuminancePeak);
}

// weighted: use the luminance weighted block mean instead of the plain mean
// unit_std: use sigma = 1 for every block (luminance weighted grey world)
BlockStats block_statistics(const rgbimage_t* img, int subwidth, int subheight,
                            bool weighted, bool unit_std) {
  BlockStats st;
  const int W = img->width, H = img->height;
  const unsigned char* ch[3] = {img->r, img->g, img->b};
  for (int by = 0; by < H; by += subheight) {
    const int y1 = std::min(by + subheight, H);
    for (int bx = 0; bx < W; bx += subwidth) {
      const int x1 = std::min(bx + subwidth, W);
      const double n = double(y1 - by) * double(x1 - bx);
      double sum[3] = {0, 0, 0}, wsum[3] = {0, 0, 0}, wtot = 0.0;
      for (int y = by; y < y1; ++y) {
        const int row = y * W;
        for (int x = bx; x < x1; ++x) {
          const unsigned char r = img->r[row + x];
          const unsigned char g = img->g[row + x];
          const unsigned char b = img->b[row + x];
          sum[0] += r; sum[1] += g; sum[2] += b;
          if (weighted) {
            const double w = triangular_weight(luminance(r, g, b));
            wtot += w;
            wsum[0] += w * r; wsum[1] += w * g; wsum[2] += w * b;
          }
        }
      }
      double mean[3];
      for (int c = 0; c < 3; ++c) {
        mean[c] = sum[c] / n;
        // a block without any weight (all black / all white) falls back to
        // its plain mean instead of producing NaN
        st.mean[c].push_back(weighted && wtot > 0.0 ? wsum[c] / wtot : mean[c]);
      }
      double sq[3] = {0, 0, 0};
      if (!unit_std) {
        for (int c = 0; c < 3; ++c) {
          for (int y = by; y < y1; ++y) {
            const int row = y * W;
            for (int x = bx; x < x1; ++x) {
              const double d = ch[c][row + x] - mean[c];
              sq[c] += d * d;
            }
          }
        }
      }
      for (int c = 0; c < 3; ++c) {
        const double s = unit_std ? 1.0 : (n > 1.0 ? sqrt(sq[c] / (n - 1.0)) : 0.0);
        st.std[c].push_back(s);
        st.std_sum[c] += s;
      }
    }
  }
  return st;
}

double* gains_from_statistics(const BlockStats& st) {
  double* gains = new double[3]();
  double sdwa[3] = {0, 0, 0};
  const size_t nblocks = st.mean[0].size();
  for (int c = 0; c < 3; ++c) {
    if (st.std_sum[c] > 0.0) {
      for (size_t i = 0; i < nblocks; ++i)
        sdwa[c] += st.std[c][i] / st.std_sum[c] * st.mean[c][i];
    } else {
      // every block is flat: weights are undefined, use the plain average
      for (size_t i = 0; i < nblocks; ++i) sdwa[c] += st.mean[c][i];
      sdwa[c] /= double(nblocks);
    }
  }
  const double avg = (sdwa[0] + sdwa[1] + sdwa[2]) / 3.0;
  for (int c = 0; c < 3; ++c) gains[c] = sdwa[c] > 0.0 ? avg / sdwa[c] : 1.0;
  return gains;
}

}  // namespace

double* calc_sdwgw(rgbimage_t* img, int subwidth, int subheight) {
  return gains_from_statistics(block_statistics(img, subwidth, subheight, false, false));
}

double* calc_sdlwgw(rgbimage_t* img, int subwidth, int subheight) {
  return gains_from_statistics(block_statistics(img, subwidth, subheight, true, false));
}

double* calc_lwgw(rgbimage_t* img, int subwidth, int subheight) {
  return gains_from_statistics(block_statistics(img, subwidth, subheight, true, true));
}

void delete_doubleptr(double* p) {
  delete[] p;
}

/*
 * Automatic Color Equalization
 *   A. Rizzi, C. Gatta, D. Marini, "A new algorithm for unsupervised global
 *   and local color correction", Pattern Recognition Letters 24(11), 2003.
 *
 * Chromatic/spatial adjustment:
 *   R_c(p) = sum_{j != p} r(I_c(p) - I_c(j)) / d(p, j)
 * with the saturation function r(x) = clip(slope * x, -limit, limit) and the
 * euclidean distance d.  Here the sum is normalised by sum_j limit / d(p, j)
 * so that R_c(p) lies in [-1, 1] whatever the sample set is.
 * The full sum is O(N^2); this implementation evaluates it on a random subset
 * of `samples` pixels shared by every target pixel (one of the speed-ups
 * suggested in the paper).  With a sparse subset a sample lying right next
 * to a pixel would dominate its sum (1/d is huge) and leave a visible blotch
 * around every sample point, so the distance is clamped from below to
 * `min_distance`.  Dynamic tone reproduction scaling then stretches
 * [min R_c, max R_c] linearly to [0, 255].
 */
namespace {

// chromatic/spatial adjustment on a random subset; fills score[c][p] and the
// per channel min/max.  nch is 1 for an aliased luminance-only call.
void ace_scores(const rgbimage_t* img, int samples, double slope, double limit,
                unsigned int seed, double min_distance, int nch,
                std::vector<float>* score, double* rmin, double* rmax);

}  // namespace

void calc_ace_scores(const rgbimage_t* img, int samples, double slope,
                     double limit, unsigned int seed, double min_distance,
                     float* scores) {
  const size_t N = size_t(img->width) * size_t(img->height);
  if (N == 0 || samples <= 0) return;
  const int nch = (img->g == img->r && img->b == img->r) ? 1 : 3;
  std::vector<float> score[3];
  double rmin[3], rmax[3];
  ace_scores(img, samples, slope, limit, seed, min_distance, nch, score, rmin, rmax);
  for (int c = 0; c < nch; ++c)
    std::copy(score[c].begin(), score[c].end(), scores + size_t(c) * N);
}

void calc_ace(rgbimage_t* img, int samples, double slope, double limit,
              unsigned int seed, double min_distance) {
  const int W = img->width, H = img->height;
  const size_t N = size_t(W) * size_t(H);
  if (N == 0 || samples <= 0) return;
  unsigned char* ch[3] = {img->r, img->g, img->b};
  const int nch = (img->g == img->r && img->b == img->r) ? 1 : 3;
  std::vector<float> score[3];
  double rmin[3], rmax[3];
  ace_scores(img, samples, slope, limit, seed, min_distance, nch, score, rmin, rmax);

  // dynamic tone reproduction scaling: [min, max] -> [0, 255]
  for (int c = 0; c < nch; ++c) {
    const double range = rmax[c] - rmin[c];
    if (range <= 0.0) {
      // flat channel: no contrast to stretch, map to middle grey
      for (size_t p = 0; p < N; ++p) ch[c][p] = 128;
      continue;
    }
    const double scale = 255.0 / range;
    for (size_t p = 0; p < N; ++p) {
      double v = (score[c][p] - rmin[c]) * scale + 0.5;
      if (v < 0.0) v = 0.0;
      if (v > 255.0) v = 255.0;
      ch[c][p] = (unsigned char)v;
    }
  }
}

namespace {

void ace_scores(const rgbimage_t* img, int samples, double slope, double limit,
                unsigned int seed, double min_distance, int nch,
                std::vector<float>* score, double* rmin, double* rmax) {
  const int W = img->width, H = img->height;
  const size_t N = size_t(W) * size_t(H);

  // random subset of pixel coordinates / values
  std::mt19937 rng(seed);
  std::vector<int> sx(samples), sy(samples);
  std::vector<int> sv[3];
  for (int c = 0; c < 3; ++c) sv[c].resize(samples);
  const unsigned char* ch[3] = {img->r, img->g, img->b};
  for (int s = 0; s < samples; ++s) {
    sx[s] = int(rng() % (unsigned int)W);
    sy[s] = int(rng() % (unsigned int)H);
    for (int c = 0; c < nch; ++c) sv[c][s] = ch[c][sy[s] * W + sx[s]];
  }

  // saturation function as a lookup table over the difference -255..255,
  // already divided by `limit`
  float sat[511];
  for (int d = -255; d <= 255; ++d) {
    double v = d * slope;
    if (v > limit) v = limit;
    if (v < -limit) v = -limit;
    sat[d + 255] = float(v / limit);
  }

  for (int c = 0; c < nch; ++c) {
    score[c].assign(N, 0.0f);
    rmax[c] = -DBL_MAX;
    rmin[c] = DBL_MAX;
  }

  std::vector<float> invd(samples);
  const float min_d2 = float(min_distance > 0.0 ? min_distance * min_distance : 0.0);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t p = size_t(y) * W + x;
      // 1/max(d(p, j), min_distance) for every sample; the pixel itself
      // (d = 0) is excluded
      double denom = 0.0;
      for (int s = 0; s < samples; ++s) {
        const int dx = x - sx[s], dy = y - sy[s];
        const int d2 = dx * dx + dy * dy;
        const float w = d2 > 0 ? 1.0f / sqrtf(std::max(float(d2), min_d2)) : 0.0f;
        invd[s] = w;
        denom += w;
      }
      if (denom <= 0.0) {  // every sample is this very pixel
        for (int c = 0; c < nch; ++c) score[c][p] = 0.0f;
        continue;
      }
      for (int c = 0; c < nch; ++c) {
        const int v = ch[c][p];
        const int* vals = &sv[c][0];
        double acc = 0.0;
        for (int s = 0; s < samples; ++s) acc += sat[v - vals[s] + 255] * invd[s];
        const double r = acc / denom;
        score[c][p] = float(r);
        if (r > rmax[c]) rmax[c] = r;
        if (r < rmin[c]) rmin[c] = r;
      }
    }
  }
}

}  // namespace

/*
 * Random Spray Retinex
 *   E. Provenzi, M. Fierro, A. Rizzi, L. De Carli, D. Gadia, D. Marini,
 *   "Random Spray Retinex", IEEE Trans. Image Processing 16(1), 2007.
 * RACE / sigma-RACE
 *   E. Provenzi, C. Gatta, M. Fierro, A. Rizzi, "A Spatially Variant
 *   White-Patch and Gray-World Method for Color Image Enhancement Driven by
 *   Local Contrast", IEEE Trans. PAMI 30(10), 2008.
 *
 * The Retinex paths are replaced by "sprays": for a target pixel x, n points
 * are drawn around it with a radial density proportional to 1/r (r uniform
 * in [0, R], angle uniform), points falling outside the image are discarded.
 *
 *   RSR  (2007, eq. 6):  L(i) = 1/N sum_k I(i) / max_{j in S_k(i)} I(j)
 *   RACE (2008, eq. 13): L(i) = 1/N sum_k 1/2 ( I(i)/I(H_k)
 *                               + 1/n~_k sum_{j in S_k(i)\{i}} s~_a(I(i)-I(j)) )
 *   with s~_a(d) = 0 for d <= -1/(2a), 1/2 + a d for |d| < 1/(2a), 1 above
 *   sigma-RACE (eq. 23): every spray term is mixed with the original value,
 *     (2 sigma_k(i))^(sigma_min/sigma_max) L_k + (1 - ...) I(i),
 *   sigma_k(i) being the standard deviation of the intensities in the spray.
 * Intensities are normalised to [0, 1] and every channel is processed on
 * its own.  Offsets come from a large precomputed pool (one random start
 * index per pixel and spray) so that every pixel gets its own sprays without
 * a trigonometric RNG per point.
 */
namespace {

struct SprayOffsets {
  std::vector<int> dx, dy;
};

enum SprayMode { SPRAY_RSR = 0, SPRAY_RACE = 1 };

struct SprayParams {
  SprayMode mode;
  int sprays, points;
  double alpha;        // RACE slope
  bool regularize;     // sigma-RACE
  double gamma;        // sigma_min / sigma_max exponent of the regulariser
};

void spray_rows(const rgbimage_t* in, rgbimage_t* out, int y0, int y1,
                const SprayParams& prm, const SprayOffsets& pool,
                unsigned int seed) {
  const int W = in->width, H = in->height;
  const unsigned char* ich[3] = {in->r, in->g, in->b};
  unsigned char* och[3] = {out->r, out->g, out->b};
  const size_t pool_size = pool.dx.size();
  std::vector<size_t> members;
  members.reserve(prm.points + 1);
  for (int y = y0; y < y1; ++y) {
    // one generator per row so the result does not depend on the threading
    std::mt19937 rng(seed + unsigned(y));
    for (int x = 0; x < W; ++x) {
      const size_t p = size_t(y) * W + x;
      double acc[3] = {0.0, 0.0, 0.0};
      for (int s = 0; s < prm.sprays; ++s) {
        size_t idx = rng() % pool_size;
        // collect the spray members inside the image (the pixel itself
        // always belongs to its spray)
        members.clear();
        for (int k = 0; k < prm.points; ++k) {
          const int xx = x + pool.dx[idx], yy = y + pool.dy[idx];
          if (++idx == pool_size) idx = 0;
          if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
          members.push_back(size_t(yy) * W + xx);
        }
        // one pass over the members for all three channels (one gather per
        // member instead of three)
        float v[3], mx[3], sum[3], sumsq[3], ace[3] = {0.0f, 0.0f, 0.0f};
        for (int c = 0; c < 3; ++c) {
          v[c] = ich[c][p] * (1.0f / 255.0f);
          mx[c] = v[c];
          sum[c] = v[c];
          sumsq[c] = v[c] * v[c];
        }
        const float alpha = float(prm.alpha), half = 0.5f / alpha;
        for (size_t m = 0; m < members.size(); ++m) {
          const size_t q = members[m];
          for (int c = 0; c < 3; ++c) {
            const float u = ich[c][q] * (1.0f / 255.0f);
            mx[c] = std::max(mx[c], u);
            if (prm.mode == SPRAY_RACE) {
              const float d = v[c] - u;
              ace[c] += d <= -half ? 0.0f : (d >= half ? 1.0f : 0.5f + alpha * d);
              sum[c] += u;
              sumsq[c] += u * u;
            }
          }
        }
        for (int c = 0; c < 3; ++c) {
          double l;
          if (prm.mode == SPRAY_RSR) {
            l = mx[c] > 0.0f ? double(v[c]) / mx[c] : 1.0;
          } else {
            const double wp = mx[c] > 0.0f ? double(v[c]) / mx[c] : 1.0;
            const double gw = members.empty() ? 0.5 : double(ace[c]) / double(members.size());
            l = 0.5 * (wp + gw);
            if (prm.regularize) {
              const double n = double(members.size() + 1);
              const double mean = sum[c] / n;
              double var = sumsq[c] / n - mean * mean;
              if (var < 0.0) var = 0.0;
              const double sigma = sqrt(var);                 // in [0, 1/2]
              const double beta = pow(std::min(2.0 * sigma, 1.0), prm.gamma);
              l = beta * l + (1.0 - beta) * v[c];
            }
          }
          acc[c] += l;
        }
      }
      for (int c = 0; c < 3; ++c) {
        double v = acc[c] / prm.sprays * 255.0 + 0.5;
        if (v > 255.0) v = 255.0;
        if (v < 0.0) v = 0.0;
        och[c][p] = (unsigned char)v;
      }
    }
  }
}

void run_sprays(const rgbimage_t* in, rgbimage_t* out, SprayParams prm,
                double radius, unsigned int seed, int threads) {
  const int W = in->width, H = in->height;
  if (W <= 0 || H <= 0 || prm.sprays <= 0 || prm.points <= 0) return;
  if (radius <= 0.0) radius = sqrt(double(W) * W + double(H) * H);

  // pool of spray offsets with radial density 1/r
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const size_t pool_size = std::max<size_t>(size_t(prm.points) * 64, 1u << 16);
  SprayOffsets pool;
  pool.dx.resize(pool_size);
  pool.dy.resize(pool_size);
  const double two_pi = 6.283185307179586;
  for (size_t i = 0; i < pool_size; ++i) {
    const double r = radius * uni(rng);
    const double t = two_pi * uni(rng);
    pool.dx[i] = int(floor(r * cos(t) + 0.5));
    pool.dy[i] = int(floor(r * sin(t) + 0.5));
  }

  if (threads <= 0) threads = int(std::thread::hardware_concurrency());
  if (threads <= 0) threads = 1;
  threads = std::min(threads, H);
  if (threads == 1) {
    spray_rows(in, out, 0, H, prm, pool, seed + 1);
    return;
  }
  std::vector<std::thread> pool_threads;
  for (int t = 0; t < threads; ++t) {
    const int y0 = int(size_t(H) * t / threads), y1 = int(size_t(H) * (t + 1) / threads);
    pool_threads.emplace_back(spray_rows, in, out, y0, y1, prm, std::cref(pool), seed + 1);
  }
  for (auto& th : pool_threads) th.join();
}

// sigma_min / sigma_max of the global channel standard deviations
double global_contrast_ratio(const rgbimage_t* in) {
  const size_t N = size_t(in->width) * size_t(in->height);
  const unsigned char* ch[3] = {in->r, in->g, in->b};
  double smin = DBL_MAX, smax = 0.0;
  for (int c = 0; c < 3; ++c) {
    double sum = 0.0, sumsq = 0.0;
    for (size_t p = 0; p < N; ++p) {
      const double v = ch[c][p] / 255.0;
      sum += v;
      sumsq += v * v;
    }
    const double mean = sum / double(N);
    double var = sumsq / double(N) - mean * mean;
    if (var < 0.0) var = 0.0;
    const double sd = sqrt(var);
    smin = std::min(smin, sd);
    smax = std::max(smax, sd);
  }
  return smax > 0.0 ? smin / smax : 1.0;
}

}  // namespace

void calc_rsr(const rgbimage_t* in, rgbimage_t* out, int sprays, int points,
              double radius, unsigned int seed, int threads) {
  SprayParams prm = {SPRAY_RSR, sprays, points, 0.0, false, 1.0};
  run_sprays(in, out, prm, radius, seed, threads);
}

void calc_race(const rgbimage_t* in, rgbimage_t* out, int sprays, int points,
               double radius, double alpha, int regularize, unsigned int seed,
               int threads) {
  SprayParams prm = {SPRAY_RACE, sprays, points, alpha, regularize != 0,
                     regularize != 0 ? global_contrast_ratio(in) : 1.0};
  run_sprays(in, out, prm, radius, seed, threads);
}
