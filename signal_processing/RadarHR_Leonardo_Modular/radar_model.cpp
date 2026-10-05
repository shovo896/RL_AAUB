#include "radar_model.h"

static float readFloatFromFlash(const float *address)
{
  return pgm_read_float(address);
}

static int8_t readInt8FromFlash(const int8_t *address)
{
  return (int8_t)pgm_read_byte(address);
}

float predictHR(const float features[N_FEATURES])
{
  float normalized[N_FEATURES];
  float hidden[N_HIDDEN];

  for (int i = 0; i < N_FEATURES; i++)
  {
    float meanValue = readFloatFromFlash(&FEATURE_MEAN[i]);
    float scaleValue = readFloatFromFlash(&FEATURE_SCALE[i]);

    if (scaleValue < 0.00000001f)
    {
      scaleValue = 1.0f;
    }

    normalized[i] = (features[i] - meanValue) / scaleValue;
  }

  for (int j = 0; j < N_HIDDEN; j++)
  {
    float sum = readFloatFromFlash(&BIAS1[j]);

    for (int i = 0; i < N_FEATURES; i++)
    {
      int index = i * N_HIDDEN + j;
      int8_t quantizedWeight = readInt8FromFlash(&W1[index]);
      float weight = ((float)quantizedWeight) * WEIGHT_SCALE;
      sum += normalized[i] * weight;
    }

    if (sum < 0.0f)
    {
      sum = 0.0f;
    }

    hidden[j] = sum;
  }

  float output = readFloatFromFlash(&BIAS2[0]);

  for (int j = 0; j < N_HIDDEN; j++)
  {
    int8_t quantizedWeight = readInt8FromFlash(&W2[j]);
    float weight = ((float)quantizedWeight) * WEIGHT_SCALE;
    output += hidden[j] * weight;
  }

  return output;
}
