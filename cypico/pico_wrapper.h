#pragma once
#include <math.h>
#include "pico/runtime/picort.h"

#ifndef MIN
    #define MIN(a, b) ((a)<(b)?(a):(b))
#endif

static char FACE_CASCADES[] =
{
    #include "pico/runtime/cascades/facefinder.ea"
};


int pico_detect_objects(const unsigned char* image, const int height,
                        const int width, const int width_step,
                        const char* cascades, const int max_detections,
                        const int n_orientations, const float scale_factor,
                        const float stride_factor, const float min_size,
                        const float q_cutoff,
                        float* qs, float* rs, float* cs, float* ss);


int pico_detect_faces(const unsigned char* image, const int height,
                      const int width, const int width_step,
                      const int max_detections,
                      const int n_orientations, const float scale_factor,
                      const float stride_factor, const float min_size,
                      const float q_cutoff,
                      float* qs, float* rs, float* cs, float* ss);
