#include <Arduino.h>
#include <avr/pgmspace.h>

/*
 ============================================================
  24 GHz Radar TinyML - Arduino Leonardo
  Model Architecture: 6 -> 4 -> 1
  INT8 Quantized Weights
  Single .ino file
 ============================================================
*/

#define N_FEATURES 6
#define N_HIDDEN 4


// ============================================================
// FEATURE NORMALIZATION
// ============================================================

static const float FEATURE_MEAN[N_FEATURES] PROGMEM = {
  169702.859722,
  45841.958767,
  1.010436,
  1.443781,
  -0.000004,
  0.016591
};


static const float FEATURE_SCALE[N_FEATURES] PROGMEM = {
  16351.694736,
  9577.646590,
  0.284819,
  0.150287,
  0.000359,
  0.038580
};


// ============================================================
// FIRST LAYER WEIGHTS
// Shape = 6 x 4
// ============================================================

static const int8_t W1[N_FEATURES * N_HIDDEN] PROGMEM = {

  -107,   0,  127,   0,

     1,  -1,  -35,  11,

   -30,  -1,  -16,   6,

    27,   0,  -54,  -1,

   -37,   0,  -79,   0,

   -87,   0,  -63,  -5
};


// ============================================================
// FIRST LAYER BIAS
//
// IMPORTANT:
// নাম B1 রাখা হয়নি, কারণ Arduino binary.h-তে B1 macro আছে
// ============================================================

static const float BIAS1[N_HIDDEN] PROGMEM = {

   4.990365,
  -0.074027,
   4.270827,
  -0.452020
};


// ============================================================
// OUTPUT LAYER WEIGHTS
// ============================================================

static const int8_t W2[N_HIDDEN] PROGMEM = {

   100,
   -14,
    93,
    -9
};


// ============================================================
// OUTPUT BIAS
// ============================================================

static const float BIAS2[1] PROGMEM = {

  2.716748
};


// ============================================================
// QUANTIZATION SCALE
// ============================================================

static const float WEIGHT_SCALE =
  0.04892230972530335f;


// ============================================================
// READ FLOAT FROM FLASH
// ============================================================

float readFloatFromFlash(const float *address)
{
  return pgm_read_float(address);
}


// ============================================================
// READ INT8 FROM FLASH
// ============================================================

int8_t readInt8FromFlash(const int8_t *address)
{
  return (int8_t)pgm_read_byte(address);
}


// ============================================================
// ML PREDICTION FUNCTION
//
// Architecture:
// 6 input
//   ↓
// 4 hidden + ReLU
//   ↓
// 1 output
// ============================================================

float predictHR(const float features[N_FEATURES])
{

  float normalized[N_FEATURES];

  float hidden[N_HIDDEN];


  // ==========================================================
  // STEP 1 — NORMALIZATION
  // ==========================================================

  for (int i = 0; i < N_FEATURES; i++)
  {

    float meanValue =
      readFloatFromFlash(
        &FEATURE_MEAN[i]
      );


    float scaleValue =
      readFloatFromFlash(
        &FEATURE_SCALE[i]
      );


    if (scaleValue < 0.00000001f)
    {
      scaleValue = 1.0f;
    }


    normalized[i] =
      (features[i] - meanValue)
      / scaleValue;
  }


  // ==========================================================
  // STEP 2 — HIDDEN LAYER
  // ==========================================================

  for (int j = 0; j < N_HIDDEN; j++)
  {

    float sum =
      readFloatFromFlash(
        &BIAS1[j]
      );


    for (int i = 0; i < N_FEATURES; i++)
    {

      int index =
        i * N_HIDDEN + j;


      int8_t quantizedWeight =
        readInt8FromFlash(
          &W1[index]
        );


      float weight =
        ((float)quantizedWeight)
        * WEIGHT_SCALE;


      sum +=
        normalized[i] * weight;
    }


    // ReLU
    if (sum < 0.0f)
    {
      sum = 0.0f;
    }


    hidden[j] = sum;
  }


  // ==========================================================
  // STEP 3 — OUTPUT LAYER
  // ==========================================================

  float output =
    readFloatFromFlash(
      &BIAS2[0]
    );


  for (int j = 0; j < N_HIDDEN; j++)
  {

    int8_t quantizedWeight =
      readInt8FromFlash(
        &W2[j]
      );


    float weight =
      ((float)quantizedWeight)
      * WEIGHT_SCALE;


    output +=
      hidden[j] * weight;
  }


  return output;
}


// ============================================================
// SETUP
// ============================================================

void setup()
{

  Serial.begin(115200);


  // Leonardo native USB
  // max 3 seconds wait

  unsigned long startTime =
    millis();


  while (
    !Serial &&
    (millis() - startTime < 3000)
  )
  {
    ;
  }


  Serial.println();

  Serial.println(
    F("========================================")
  );

  Serial.println(
    F("24 GHz Radar TinyML")
  );

  Serial.println(
    F("Arduino Leonardo")
  );

  Serial.println(
    F("INT8 Tiny Neural Network")
  );

  Serial.println(
    F("Architecture: 6 -> 4 -> 1")
  );

  Serial.println(
    F("========================================")
  );


  // ==========================================================
  // TEST INPUT
  //
  // আপাতত test feature
  // পরে live radar features আসবে
  // ==========================================================

  float features[N_FEATURES] = {

    169702.859722f,

    45841.958767f,

    1.010436f,

    1.443781f,

    -0.000004f,

    0.016591f
  };


  Serial.println();

  Serial.println(
    F("Running inference...")
  );


  // ==========================================================
  // RUN ML MODEL
  // ==========================================================

  float predictedHR =
    predictHR(features);


  // ==========================================================
  // PRINT RESULT
  // ==========================================================

  Serial.println();


  Serial.print(
    F("Predicted Heart Rate = ")
  );


  Serial.print(
    predictedHR,
    2
  );


  Serial.println(
    F(" BPM")
  );


  Serial.println();


  Serial.println(
    F("Inference completed.")
  );


  Serial.println(
    F("========================================")
  );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{

  // এখন empty
  //
  // পরের step:
  //
  // Radar I/Q
  //    ↓
  // AnalogRead
  //    ↓
  // Feature extraction
  //    ↓
  // 6 features
  //    ↓
  // predictHR()
  //    ↓
  // Heart Rate

}