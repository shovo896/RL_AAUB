#include <Arduino.h>

#include "radar_model.h"

void setup()
{
  Serial.begin(115200);

  unsigned long startTime = millis();

  while (!Serial && (millis() - startTime < 3000))
  {
    ;
  }

  Serial.println();
  Serial.println(F("========================================"));
  Serial.println(F("24 GHz Radar TinyML"));
  Serial.println(F("Arduino Leonardo"));
  Serial.println(F("INT8 Tiny Neural Network"));
  Serial.println(F("Architecture: 6 -> 4 -> 1"));
  Serial.println(F("========================================"));

  float features[N_FEATURES] = {
    169702.859722f,
    45841.958767f,
    1.010436f,
    1.443781f,
    -0.000004f,
    0.016591f
  };

  Serial.println();
  Serial.println(F("Running inference..."));

  float predictedHR = predictHR(features);

  Serial.println();
  Serial.print(F("Predicted Heart Rate = "));
  Serial.print(predictedHR, 2);
  Serial.println(F(" BPM"));
  Serial.println();
  Serial.println(F("Inference completed."));
  Serial.println(F("========================================"));
}

void loop()
{
}
