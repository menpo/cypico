#
#	Copyright (c) 2013, Nenad Markus
#	All rights reserved.
#
#	This is an implementation of the algorithm described in the following paper:
#		N. Markus, M. Frljak, I. S. Pandzic, J. Ahlberg and R. Forchheimer,
#		A method for object detection based on pixel intensity comparisons,
#		http://arxiv.org/abs/1305.4537
#
#	Redistribution and use of this program as source code or in binary form,
#   with or without modifications, are permitted provided that the following
#   conditions are met:
#		1. Redistributions may not be sold, nor may they be used in a commercial
#          product or activity without prior permission from the copyright
#          holder (contact him at nenad.markus@fer.hr).
#		2. Redistributions may not be used for military purposes.
#		3. Any published work which utilizes this program shall include the
#          reference to the paper available at http://arxiv.org/abs/1305.4537
#		4. Redistributions must retain the above copyright notice and the
#          reference to the algorithm on which the implementation is based on,
#          this list of conditions and the following disclaimer.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#   USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The above copyright notice is copied from the original Pico source, which
# is distributed with this package. Note that this particular Cython file,
# pico.pyx, is separately licensed from the main Pico project (BSD 3-clause)
# which can be found in the main repository under LICENSE.md

# distutils: include_dirs = ./
# distutils: sources = pico/runtime/picort.c cypico/pico_wrapper.c
import numpy as np
cimport numpy as np


cdef extern from "pico_wrapper.h":
    int pico_detect_faces(const unsigned char* image, const int height,
                          const int width, const int width_step,
                          const int max_detections,
                          const int n_orientations, const float scale_factor,
                          const float stride_factor, const float min_size,
                          const float q_cutoff,
                          float* qs, float* rs, float* cs, float* ss)


cpdef detect_faces(unsigned char[:, :] image, int max_detections=100,
                   int n_orientations=1, float scale_factor=1.2,
                   float stride_factor=0.1, float min_size=100,
                   float confidence_cutoff=3.0):
    r"""
    Detect faces in the given image. It will detect multiple faces and has
    the ability to detect faces that have been *in-plane* rotated. This implies
    upside down faces, but not faces turned profile.

    Parameters
    ----------
    image : unsigned char[:, :]
        The image to detect faces within. Should be a `uint8` image with values
        in the range 0 to 255.
    max_detections : int
        The maximum number of detections to return.
    n_orientations : int
        The number of orientations of the cascades to use. 1 will perform an
        axis aligned detection. Values greater than 1 will perform
        rotations of the cascade equally around a unit circle (equal number
        of degrees).
    scale_factor : float
        The ratio to increase the cascade window at every iteration. Must
        be great than 1.0
    stride_factor : float
        The ratio to decrease the window step by at every iteration. Must be
        less than 1.0
    min_size : float
        The minimum size in pixels (diameter of the detection circle) that a
        face can be. This is the starting cascade window size.
    confidence_cutoff : float
        The confidence value to trim the detections with. Any detections with
        confidence less than the cutoff will be discarded.

    Returns
    -------
    confidences : (n_detections,) ndarray
        The list of confidences of the detector that were greater than
        the provided confidence cutoff.
    y_coords : (n_detections,) ndarray
        The list of floating point y-coordinates (row indices). This
        represents the x-coordinate of the centre of the circle the face is
        inside.
    x_coords : (n_detections,) ndarray
        The list of floating point x-coordinates (column indices). This
        represents the x-coordinate of the centre of the circle the face is
        inside.
    scales : (n_detections,) ndarray
        The list of floating point scales of the detections. This represents
        the diameter of the circle the face is inside.

    Raises
    ------
    ValueError:
        If scale_factor is less than or equal to 1.0
        If stride_factor is greater than or equal to 1.0
    """
    if scale_factor <= 1.0:
        raise ValueError('Scale factor must be greater than 1.0')
    if stride_factor >= 1.0:
        raise ValueError('Scale factor must be less than 1.0')
    cdef:
        int height = image.shape[0]
        int width = image.shape[1]
        int n_detections = 0
        float[:] confidences = np.zeros(max_detections, dtype=np.float32)
        float[:] y_coords    = np.zeros(max_detections, dtype=np.float32)
        float[:] x_coords    = np.zeros(max_detections, dtype=np.float32)
        float[:] scales      = np.zeros(max_detections, dtype=np.float32)

    # Call the c wrapper that I created that hard codes the provided
    n_detections = pico_detect_faces(&image[0, 0], height, width, width,
                                     max_detections, n_orientations,
                                     scale_factor, stride_factor,
                                     min_size, confidence_cutoff,
                                     &confidences[0], &y_coords[0],
                                     &x_coords[0], &scales[0])

    return (np.resize(confidences, n_detections),
            np.resize(y_coords, n_detections),
            np.resize(x_coords, n_detections),
            np.resize(scales, n_detections))
