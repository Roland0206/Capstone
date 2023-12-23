/*
  steerableDetector: steerable filters for edge and ridge detection

  References:
    [1] Jacob and Unser, IEEE Trans. Pattern Anal. Mach. Intell. 26(8), 2004.

  Copyright (c) 2005-2017 Francois Aguet

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef STEERABLEFILTER2D_H_
#define STEERABLEFILTER2D_H_


namespace steerable {

#define PI 3.141592653589793
#define TOL 1e-12


class SteerableDetector {

public:
    SteerableDetector(const double *pixels, const size_t nx, const size_t ny, const int order, const double sigma, const int borderCondition);
    SteerableDetector(const double *pixels, const size_t nx, const size_t ny, const int order, const double sigma);
    ~SteerableDetector();
    void setOrder(const int M);
    void filter();
    void filter(const double sigma);
    void runNMS();
    void getAngleResponse(double* p, const size_t nt);
    void getAngleResponseWithArray(double* p, const double* angles, const size_t nAngles);

    const double *pixels_;
    size_t nx_, ny_;
    int M_;
    double sigma_;  // standard deviation
    double window_;  // kernel size: [-window*sigma ... window*sigma]
    int borderCondition_;

    double *response_, *orientation_, *nms_response_;

private:
    typedef void (SteerableDetector::*filterFct)();
    typedef double (SteerableDetector::*pointrespFct)(int, double);

    void init(const double *pixels, const size_t nx, const size_t ny, const int order, const double sigma, const int borderCondition);

    filterFct filter_fct_;
    pointrespFct pointresp_fct_;

    double **templates_;
    int nTemplates_;
    double *alpha_;

    static int getNumTemplates(const int M);
    void computeWeights();
    void computeTemplates();

    // Point responses
    double pointRespM1(const int i, const double angle);
    double pointRespM2(const int i, const double angle);
    double pointRespM3(const int i, const double angle);
    double pointRespM4(const int i, const double angle);
    double pointRespM5(const int i, const double angle);

    void filterM1();
    void filterM2();
    void filterM3();
    void filterM4();
    void filterM5();

    static int mirror(const int x, const int nx);
    static double interp(const double* image, const int nx, const int ny, const double x, const double y);
    static double opposite(const double theta);
    static double approxZero(const double n);
};

}  // namespace steerable

#endif // STEERABLEFILTER2D_H_
