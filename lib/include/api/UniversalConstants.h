#ifndef UNIVERSALCONSTANTS_H
#define UNIVERSALCONSTANTS_H
#define _USE_MATH_DEFINES
#include <math.h>

#ifdef __cplusplus
    extern "C" {
#endif

typedef struct {
    //gravitational constant
    float G;
    //speed of light: 299792458
    float C;
} UniversalConstants;

const UniversalConstants real_universal_constants = {
    .G = 6.6743e-11,
    .C = 299792458.0
};

//multiply by this to convert radians to degrees
const float RAD2DEG = 180/M_PI;
//multiply by this to convert degrees to radians 
const float DEG2RAD = M_PI/180;

#ifdef __cplusplus
    }
#endif

#endif