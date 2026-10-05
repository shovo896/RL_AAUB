#pragma once

#include <Arduino.h>
#include <avr/pgmspace.h>

constexpr uint8_t N_FEATURES = 6;
constexpr uint8_t N_HIDDEN = 4;

extern const float FEATURE_MEAN[N_FEATURES];
extern const float FEATURE_SCALE[N_FEATURES];
extern const int8_t W1[N_FEATURES * N_HIDDEN];
extern const float BIAS1[N_HIDDEN];
extern const int8_t W2[N_HIDDEN];
extern const float BIAS2[1];
extern const float WEIGHT_SCALE;
