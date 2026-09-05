#include <Arduino.h>
#include <avr/pgmspace.h>

#include "radar_model.h"

void setup() {

    Serial.begin(115200);

    while (!Serial) {
        ;
    }

    Serial.println("Radar TinyML Model Loaded!");
}

void loop() {

}