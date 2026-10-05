#include "model_params.h"

const float FEATURE_MEAN[N_FEATURES] PROGMEM = {
  169702.859722,
  45841.958767,
  1.010436,
  1.443781,
  -0.000004,
  0.016591
};

const float FEATURE_SCALE[N_FEATURES] PROGMEM = {
  16351.694736,
  9577.646590,
  0.284819,
  0.150287,
  0.000359,
  0.038580
};

const int8_t W1[N_FEATURES * N_HIDDEN] PROGMEM = {
  -107,   0,  127,   0,
     1,  -1,  -35,  11,
   -30,  -1,  -16,   6,
    27,   0,  -54,  -1,
   -37,   0,  -79,   0,
    -87,   0,  -63,  -5
};

const float BIAS1[N_HIDDEN] PROGMEM = {
   4.990365,
  -0.074027,
   4.270827,
  -0.452020
};

const int8_t W2[N_HIDDEN] PROGMEM = {
   100,
   -14,
    93,
    -9
};

const float BIAS2[1] PROGMEM = {
  2.716748
};

const float WEIGHT_SCALE = 0.04892230972530335f;
