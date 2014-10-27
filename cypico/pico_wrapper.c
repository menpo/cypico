#include "pico_wrapper.h"


int pico_detect_objects(const unsigned char* image, const int height,
                        const int width, const int width_step,
                        const char* cascades, const int max_detections,
                        const int n_orientations, const float scale_factor,
                        const float stride_factor, const float min_size,
                        const float q_cutoff,
                        float* qs, float* rs, float* cs, float* ss) {
    int n_detections = 0;
    int i = 0;
    float orientation = 0.0f;

    if (n_orientations == 1) {
        n_detections = find_objects(0.0f, rs, cs, ss, qs, max_detections,
                                    cascades, image, height, width,
                                    width_step, scale_factor, stride_factor,
                                    min_size, MIN(height, width), 1);
    } else {
        // Scan the image at n_orientations different orientations
        for(i = 0; i < n_orientations; i++) {
            orientation = i * 2.0 * M_PI / n_orientations;
            n_detections += find_objects(orientation,
                                         &rs[n_detections], &cs[n_detections],
                                         &ss[n_detections], &qs[n_detections],
                                         max_detections - n_detections,
                                         cascades, image, height, width,
                                         width_step, scale_factor,
                                         stride_factor, min_size,
                                         MIN(height, width), 1);
        }
    }

    // Given detections of low confidence, we remove them
    for(i = n_detections - 1; i > 0; i--) {
    	if (qs[i] < q_cutoff) {
            qs[i] = qs[n_detections - 1];
            rs[i] = rs[n_detections - 1];
            cs[i] = cs[n_detections - 1];
            ss[i] = ss[n_detections - 1];
            n_detections--;
    	}
    }

    return n_detections;
}


int pico_detect_faces(const unsigned char* image, const int height,
                      const int width, const int width_step,
                      const int max_detections,
                      const int n_orientations, const float scale_factor,
                      const float stride_factor, const float min_size,
                      const float q_cutoff,
                      float* qs, float* rs, float* cs, float* ss) {
   return pico_detect_objects(image, height, width, width_step,
                              FACE_CASCADES, max_detections, n_orientations,
                              scale_factor, stride_factor, min_size, q_cutoff,
                              qs, rs, cs, ss);
}
