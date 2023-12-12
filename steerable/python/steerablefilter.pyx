# distutils: language = c++
# distutils: sources = ../src/steerableDetector.cpp
# cython: language_level=3

import numpy as np
cimport numpy as np
np.import_array()
import ctypes
from matplotlib.colors import hsv_to_rgb


cdef extern from "steerableDetector.h" namespace "steerable":
    cdef cppclass SteerableDetector:
        SteerableDetector(double*, const size_t, const size_t, const int, const double, const int) except +
        void setOrder(const int)
        void filter()
        void runNMS()
        void getAngleResponse(double* p, const size_t nt)

        const double *pixels_
        size_t nx_, ny_
        int M_
        double sigma_
        int borderCondition_

        double *response_
        double *orientation_
        double *nms_response_


cdef class Detector2D:
    cdef SteerableDetector *thisptr
    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] pixels, int order, double sigma, int border_condition=3):
        if order<1 or order>5:
            raise ValueError('Order must be an integer between 1 and 5.')
        if not np.all(np.isfinite(pixels)):
            raise ValueError('Input image values must be finite.')
        if sigma<=0:
            raise ValueError('Sigma must be strictly positive.')
        L = 2*(int)(4.0*sigma)+1
        nx = pixels.shape[0]
        ny = pixels.shape[1]
        if L>nx or L>ny:
            raise ValueError('Filter support is larger than image. Sigma must be < {}'.format((np.minimum(nx,ny)-1)/8))
        if border_condition<0 or border_condition>3:
            raise ValueError('Allowed border_condition values: 0: zero-padding; 1:replicate; 2: periodic; 3: mirror.')
        self.thisptr = new SteerableDetector(&pixels[0,0], pixels.shape[0], pixels.shape[1], order, sigma, border_condition)
    def __dealloc__(self):
        del self.thisptr
    def filter(self):
        """Apply steerable filter"""
        self.thisptr.filter()
        cdef np.npy_intp dims[2]
        dims[0] = self.thisptr.nx_
        dims[1] = self.thisptr.ny_
        response = np.PyArray_SimpleNewFromData(2, &dims[0], np.NPY_FLOAT64, self.thisptr.response_)
        orientation = np.PyArray_SimpleNewFromData(2, &dims[0], np.NPY_FLOAT64, self.thisptr.orientation_)
        return response, orientation
    def get_nms(self):
        """Run and return non-maximum suppression"""
        self.thisptr.runNMS()
        cdef np.npy_intp dims[2]
        dims[0] = self.thisptr.nx_
        dims[1] = self.thisptr.ny_
        return np.PyArray_SimpleNewFromData(2, &dims[0], np.NPY_FLOAT64, self.thisptr.nms_response_)
    def get_angle_response(self, int n=36):
        """Compute filter response for a range of angles (default: 36)"""
        cdef np.ndarray[np.double_t, ndim=3, mode="c"] fb
        fb = np.zeros([n, self.thisptr.nx_, self.thisptr.ny_])
        self.thisptr.getAngleResponse(&fb[0,0,0], n)
        return fb
    def make_composite(self, response, orientation):
        """RGB composite image of filter response and orientation"""
        if self.thisptr.M_%2==0:
            a = (orientation+np.pi/2)/np.pi
        else:
            a = (orientation+np.pi)/(2*np.pi)
        rescale = lambda s: (s-np.min(s))/(np.max(s)-np.min(s))
        composite = np.dstack((a, np.ones(response.shape), rescale(response)))
        composite = (255*hsv_to_rgb(composite)).astype(np.uint8)
        return composite
