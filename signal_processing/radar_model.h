#ifndef RADAR_MODEL_H
#define RADAR_MODEL_H
#include <avr/pgmspace.h>

static const float FEATURE_MEAN[] = {169702.859722, 45841.958767, 1.010436, 1.443781, -0.000004, 0.016591};
static const float FEATURE_SCALE[] = {16351.694736, 9577.646590, 0.284819, 0.150287, 0.000359, 0.038580};
static const int8_t W1[] = {-107, 0, 127, 0, 1, -1, -35, 11, -30, -1, -16, 6, 27, 0, -54, -1, -37, 0, -79, 0, -87, 0, -63, -5};
static const float B1[] = {4.990365, -0.074027, 4.270827, -0.452020};
static const int8_t W2[] = {100, -14, 93, -9};
static const float B2[] = {2.716748};
#endif // RADAR_MODEL_H
