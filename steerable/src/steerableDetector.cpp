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

#include <string>
#include <math.h>
#include <gsl/gsl_poly.h>
#include <algorithm>

#include "convolver.h"
#include "steerableDetector.h"

namespace steerable {

void SteerableDetector::init(const double *pixels, const size_t nx, const size_t ny, const int order, const double sigma, const int borderCondition) {
    pixels_ = pixels;
    nx_ = nx;
    ny_ = ny;
    sigma_ = sigma;
    borderCondition_ = borderCondition;
    window_ = 4.0;
    
    setOrder(order);  // sets M_
    
    // allocate
    int N = nx_*ny_;
    nTemplates_ = getNumTemplates(M_);
    templates_ = new double*[nTemplates_];
    for (int i=0;i<nTemplates_;++i) {
        templates_[i] = new double[N];
    }
    response_ = new double[N];
    orientation_ = new double[N];
    nms_response_ = new double[N];
    
    // precompute filterbank templates
    computeTemplates();
    computeWeights();
}


SteerableDetector::SteerableDetector(const double *pixels, const size_t nx, const size_t ny,
    const int order, const double sigma, const int borderCondition) {
    init(pixels, nx, ny, order, sigma, borderCondition);
}


SteerableDetector::SteerableDetector(const double *pixels, const size_t nx, const size_t ny,
    const int order, const double sigma) {
    init(pixels, nx, ny, order, sigma, 3);  // mirror border
}


SteerableDetector::~SteerableDetector() {
    for (int i=0;i<nTemplates_;++i) {
        delete [] templates_[i];
    }
    delete[] templates_;
    delete[] response_;
    delete[] nms_response_;
    delete[] orientation_;
    delete[] alpha_;
}


void SteerableDetector::computeTemplates() {
    int wWidth = (int)(window_*sigma_);
    int nk = wWidth+1;
    
    double* aKernel = new double[nk];
    double* bKernel = new double[nk];
    double* g = new double[nk];
    double* buffer = new double[nx_*ny_];
    double d;
    double sigma2 = sigma_*sigma_;
    double sigma4 = sigma2*sigma2;
    double sigma6 = sigma4*sigma2;
    double sigma8 = sigma4*sigma4;
    double sigma10 = sigma6*sigma4;
    double sigma12 = sigma8*sigma4;
    
    for (int i=0;i<nk;i++) {
        g[i] = exp(-(i*i)/(2.0*sigma2));
    }
    
    // set up convolver
    convolver::Convolver conv = convolver::Convolver(pixels_, nx_, ny_, borderCondition_);
    
    if (M_ == 1 || M_ == 3 || M_ == 5) {
        d = 2.0*PI*sigma4;
        for (int i=0;i<nk;i++) {
            aKernel[i] = -i*g[i] / d;
        }
        
        conv.convolveOddXEvenY(aKernel, nk, g, nk, templates_[0]);  // g_x
        conv.convolveEvenXOddY(g, nk, aKernel, nk, templates_[1]);  // g_y
        
        if (M_ == 3 || M_ == 5) {
            d = 2.0*PI*sigma8;
            for (int i=0;i<nk;i++) {
                aKernel[i] = (3.0*i*sigma2 - i*i*i) * g[i] / d;
            }
            conv.convolveOddXEvenY(aKernel, nk, g, nk, templates_[2]);  // g_xxx
            conv.convolveEvenXOddY(g, nk, aKernel, nk, templates_[5]);  // g_yyy
            for (int i=0;i<nk;i++) {
                aKernel[i] = (sigma2 - i*i) * g[i] / d;
                bKernel[i] = i*g[i];
            }
            conv.convolveEvenXOddY(aKernel, nk, bKernel, nk, templates_[3]);  // gxxy
            conv.convolveOddXEvenY(bKernel, nk, aKernel, nk, templates_[4]);  // gxyy
        }
        if (M_ == 5) {
            d = 2.0*PI*sigma12;
            for (int i=0;i<nk;i++) {
                aKernel[i] = -i*(i*i*i*i - 10.0*i*i*sigma2 + 15.0*sigma4) * g[i] / d;
            }
            conv.convolveOddXEvenY(aKernel, nk, g, nk, templates_[6]);  // gxxxxx
            conv.convolveEvenXOddY(g, nk, aKernel, nk, templates_[11]);  // gyyyyy
            for (int i=0;i<nk;i++) {
                aKernel[i] = (i*i*i*i - 6.0*i*i*sigma2 + 3.0*sigma4) * g[i] / d;
                bKernel[i] = -i * g[i];
            }
            conv.convolveEvenXOddY(aKernel, nk, bKernel, nk, templates_[7]);  // g_xxxxy
            conv.convolveOddXEvenY(bKernel, nk, aKernel, nk, templates_[10]);  // g_xyyyy
            for (int i=0;i<nk;i++) {
                aKernel[i] = i*(i*i - 3.0*sigma2) * g[i] / d;
                bKernel[i] = (sigma2 - i*i) * g[i];
            }
            conv.convolveOddXEvenY(aKernel, nk, bKernel, nk, templates_[8]);  // g_xxxyy
            conv.convolveEvenXOddY(bKernel, nk, aKernel, nk, templates_[9]);  // g_xxyyy
        }
    } else { //(M == 2 || M == 4)
        
        d = 2.0*PI*sigma6;
        for (int i=0;i<nk;i++) {
            aKernel[i] = (i*i - sigma2) * g[i] / d;
        }
        conv.convolveEvenXEvenY(aKernel, nk, g, nk, templates_[0]);  // g_xx
        conv.convolveEvenXEvenY(g, nk, aKernel, nk, templates_[2]);  // g_yy
        for (int i=0;i<nk;i++) {
            aKernel[i] = i * g[i];
            bKernel[i] = aKernel[i] / d;
        }
        conv.convolveOddXOddY(aKernel, nk, bKernel, nk, templates_[1]);  // g_xy
        
        if (M_ == 4) {
            d = 2.0*PI*sigma10;
            for (int i=0;i<nk;i++) {
                aKernel[i] = (i*i*i*i - 6.0*i*i*sigma2 + 3.0*sigma4) * g[i] / d;
            }
            conv.convolveEvenXEvenY(aKernel, nk, g, nk, templates_[3]);  // g_xxxx
            conv.convolveEvenXEvenY(g, nk, aKernel, nk, templates_[7]);  // g_yyyy
            for (int i=0;i<nk;i++) {
                aKernel[i] = i * (i*i - 3.0*sigma2) * g[i] / d;
                bKernel[i] = i * g[i];
            }
            conv.convolveOddXOddY(aKernel, nk, bKernel, nk, templates_[4]);  // g_xxxy
            conv.convolveOddXOddY(bKernel, nk, aKernel, nk, templates_[6]);  // g_xyyy
            for (int i=0;i<nk;i++) {
                aKernel[i] = (sigma2 - i*i) * g[i];
                bKernel[i] = aKernel[i] / d;
            }
            conv.convolveEvenXEvenY(aKernel, nk, bKernel, nk, templates_[5]);  // g_xxyy
        }
    }
    
    // free memory
    delete[] aKernel;
    delete[] bKernel;
    delete[] g;
    delete[] buffer;
}


void SteerableDetector::computeWeights() {
    double s2 = sigma_*sigma_;
    double s3 = sigma_*s2;
    switch (M_) {
        case 1:
            alpha_ = new double[1];
            alpha_[0] = sqrt(2.0/PI);
            break;
        case 2: 
            alpha_ = new double[2];
            // mu = 0
            alpha_[0] = -sqrt(3.0/(4.0*PI)) * sigma_;
            alpha_[1] = -alpha_[0] / 3.0;
            // mu = 2
            //alpha_[0] = -sqrt(2.0/(3.0*PI))*sigma_;
            //alpha_[0] = -sigma_; // Hessian
            //alpha_[1] = 0.0;
            break;
        case 3: // mu = 0
            alpha_ = new double[3];
            alpha_[0] = 0.966;
            alpha_[1] = 0.0;
            alpha_[2] = 0.256*s2; 
            break;
        case 4: // mu = 0.25
            alpha_ = new double[5];
            alpha_[0] = -0.392*sigma_;
            alpha_[1] = 0.113*sigma_;
            alpha_[2] = 0.034*s3;
            alpha_[3] = -0.184*s3;
            alpha_[4] = 0.025*s3; 
            break;
        case 5:
            alpha_ = new double[5];
            alpha_[0] = 1.1215;
            alpha_[1] = 0.018*s2;
            alpha_[2] = 0.5576*s2;
            alpha_[3] = 0.0038*s2*s2;
            alpha_[4] = 0.0415*s2*s2;
            break;
        default:
            alpha_ = NULL;
    }
}


void SteerableDetector::setOrder(const int M) {
    M_ = M;
    switch (M_) {
        case 1:
            filter_fct_ = &SteerableDetector::filterM1;
            pointresp_fct_ = &SteerableDetector::pointRespM1;
            break;
        case 2:
            filter_fct_ = &SteerableDetector::filterM2;
            pointresp_fct_ = &SteerableDetector::pointRespM2;
            break;
        case 3:
            filter_fct_ = &SteerableDetector::filterM3;
            pointresp_fct_ = &SteerableDetector::pointRespM3;
            break;
        case 4:
            filter_fct_ = &SteerableDetector::filterM4;
            pointresp_fct_ = &SteerableDetector::pointRespM4;
            break;
        case 5:
            filter_fct_ = &SteerableDetector::filterM5;
            pointresp_fct_ = &SteerableDetector::pointRespM5;
            break;
    }
}


void SteerableDetector::filter() {
    (this->*filter_fct_)();
}


void SteerableDetector::filter(const double sigma) {
    sigma_ = sigma;
    computeTemplates();
    computeWeights();
    (this->*filter_fct_)();
}


void SteerableDetector::runNMS() {
    // non-maximum suppression
    double ux, uy, v1, v2;

    div_t divRes;
    for (size_t i=0;i<nx_*ny_;++i) {
        divRes = div((int)i, nx_);
        ux = cos(orientation_[i]);
        uy = sin(orientation_[i]);
        v1 = interp(response_, nx_, ny_, divRes.rem+ux, divRes.quot+uy);
        v2 = interp(response_, nx_, ny_, divRes.rem-ux, divRes.quot-uy);
        if (v1 > response_[i] || v2 > response_[i]) {
            nms_response_[i] = 0.0;
        } else {
            nms_response_[i] = response_[i];
        }
    }
}


void SteerableDetector::getAngleResponse(double* p, const size_t nt) {
    double dt;
    div_t divRes;
    int N = nx_*ny_;
    if (M_ % 2 == 0) {
        dt = PI/nt;
    } else {
        dt = 2.0*PI/nt;
    }
    for (size_t t=0;t<nt;++t) {
        for (int i=0;i<N;++i) {
            divRes = div(i, nx_);
            p[t*N + divRes.quot+divRes.rem*ny_] = (this->*pointresp_fct_)(i, t*dt);
        }
    }
}

void SteerableDetector::getAngleResponseWithArray(double* p, const double* angles, const size_t nAngles) {
    div_t divRes;
    int N = nx_*ny_;
    for (size_t t=0; t<nAngles; ++t) {
        for (int i=0; i<N; ++i) {
            divRes = div(i, nx_);
            p[t*N + divRes.quot+divRes.rem*ny_] = (this->*pointresp_fct_)(i, angles[t]);
        }
    }
}


// Point responses
double SteerableDetector::pointRespM1(const int i, const double angle) {
    double cosT = cos(angle);
    double sinT = sin(angle);
    
    double gxi = templates_[0][i];
    double gyi = templates_[1][i];
    double a11 = alpha_[0];
    
    double r = a11 * (cosT*gxi + sinT*gyi);
    return r;
}


double SteerableDetector::pointRespM2(const int i, const double angle) {
    double cosT = cos(angle);
    double sinT = sin(angle);
    
    double gxxi = templates_[0][i];
    double gxyi = templates_[1][i];
    double gyyi = templates_[2][i];
    double a20 = alpha_[0];
    double a22 = alpha_[1];
    
    double r = sinT*sinT * (a20*gyyi+a22*gxxi)
             + sinT*cosT*2.0*(a20-a22)*gxyi
             + cosT*cosT * (a20*gxxi+a22*gyyi);
    return r;
}


double SteerableDetector::pointRespM3(const int i, const double angle) {
    double cosT = cos(angle);
    double sinT = sin(angle);
    double cosT2 = cosT*cosT;
    double sinT2 = sinT*sinT;
    
    double gxi = templates_[0][i];
    double gyi = templates_[1][i];
    double gxxxi = templates_[2][i];
    double gxxyi = templates_[3][i];
    double gxyyi = templates_[4][i];
    double gyyyi = templates_[5][i];
    
    double a10 = alpha_[0];
    double a30 = alpha_[1];
    double a32 = alpha_[2];
    
    double r = a10*(sinT*gyi + cosT*gxi)
             + sinT2*sinT*(a30*gyyyi + a32*gxxyi)
             + cosT*sinT2*(a32*gxxxi + (3.0*a30-2.0*a32)*gxyyi)
             + cosT2*sinT*(a32*gyyyi + (3.0*a30-2.0*a32)*gxxyi)
             + cosT2*cosT*(a32*gxyyi + a30*gxxxi);
    return r;
}


double SteerableDetector::pointRespM4(const int i, const double angle) {
    double cosT = cos(angle);
    double sinT = sin(angle);
    double sinTcosT = cosT*sinT;
    double cosT2 = cosT*cosT;
    double sinT2 = sinT*sinT;
    
    double gxxi = templates_[0][i];
    double gxyi = templates_[1][i];
    double gyyi = templates_[2][i];
    double gxxxxi = templates_[3][i];
    double gxxxyi = templates_[4][i];
    double gxxyyi = templates_[5][i];
    double gxyyyi = templates_[6][i];
    double gyyyyi = templates_[7][i];
    
    double a20 = alpha_[0];
    double a22 = alpha_[1];
    double a40 = alpha_[2];
    double a42 = alpha_[3];
    double a44 = alpha_[4];
       
    double r = sinT2*(a20*gyyi + a22*gxxi)
             + sinT*cosT*2.0*(a20-a22)*gxyi
             + cosT2*(a20*gxxi + a22*gyyi)
             + sinT2*sinT2*(a40*gyyyyi + a42*gxxyyi + a44*gxxxxi)
             + sinT2*sinTcosT*2.0*((2.0*a40-a42)*gxyyyi + (a42-2.0*a44)*gxxxyi)
             + sinT2*cosT2*(a42*gyyyyi + (6.0*a40 - 4.0*a42 + 6.0*a44)*gxxyyi + a42*gxxxxi)
             + sinTcosT*cosT2*2.0*((2.0*a40-a42)*gxxxyi + (a42-2.0*a44)*gxyyyi)
             + cosT2*cosT2*(a44*gyyyyi + a42*gxxyyi + a40*gxxxxi);
    return r;
}


double SteerableDetector::pointRespM5(const int i, const double angle) {
    double cosT = cos(angle);
    double sinT = sin(angle);
    double cosT2 = cosT*cosT;
    double sinT2 = sinT*sinT;
    double cosT3 = cosT2*cosT;
    double sinT3 = sinT2*sinT;
    double sinT4 = sinT2*sinT2;
    double cosT4 = cosT2*cosT2;
    double sinT5 = sinT2*sinT3;
    double cosT5 = cosT2*cosT3;
    
    double gxi = templates_[0][i];
    double gyi = templates_[1][i];
    double gxxxi = templates_[2][i];
    double gxxyi = templates_[3][i];
    double gxyyi = templates_[4][i];
    double gyyyi = templates_[5][i];
    double gxxxxxi = templates_[6][i];
    double gxxxxyi = templates_[7][i];
    double gxxxyyi = templates_[8][i];
    double gxxyyyi = templates_[9][i];
    double gxyyyyi = templates_[10][i];
    double gyyyyyi = templates_[11][i];
    
    double a10 = alpha_[0];
    double a30 = alpha_[1];
    double a32 = alpha_[2];
    double a52 = alpha_[3];
    double a54 = alpha_[4];
    
    double r = a10*(sinT*gyi + cosT*gxi)
             + sinT3*(a30*gyyyi + a32*gxxyi)
             + sinT2*cosT*((3.0*a30-2.0*a32)*gxyyi + a32*gxxxi)
             + cosT2*sinT*((3.0*a30-2.0*a32)*gxxyi + a32*gyyyi)
             + cosT3*(a32*gxyyi + a30*gxxxi)
             + sinT5*(a52*gxxyyyi + a54*gxxxxyi)
             + sinT4*cosT*(-2.0*a52*gxyyyyi + (3.0*a52-4.0*a54)*gxxxyyi + a54*gxxxxxi)
             + sinT3*cosT2*(a52*gyyyyyi + 6.0*(a54-a52)*gxxyyyi + (3.0*a52-4.0*a54)*gxxxxyi)
             + sinT2*cosT3*(a52*gxxxxxi + 6.0*(a54-a52)*gxxxyyi + (3.0*a52-4.0*a54)*gxyyyyi)
             + cosT4*sinT*(-2.0*a52*gxxxxyi + (3.0*a52-4.0*a54)*gxxyyyi + a54*gyyyyyi)
             + cosT5*(a54*gxyyyyi + a52*gxxxyyi);
    return r;
}


void SteerableDetector::filterM1() {
    double* gx = templates_[0];
    double* gy = templates_[1];
    double a11 = alpha_[0];
    
    double* tRoots = new double[2];
    double gxi, gyi;
    
    for (size_t i=0;i<nx_*ny_;++i) {
        gxi = approxZero(gx[i]);
        gyi = approxZero(gy[i]);
        
        orientation_[i] = atan2(gyi,gxi);
        response_[i] = a11*sqrt(gxi*gxi + gyi*gyi);
    }
    delete[] tRoots;
}


void SteerableDetector::filterM2() {
    // quadratic root solution
    double* gxx = templates_[0];
    double* gxy = templates_[1];
    double* gyy = templates_[2];
    double a20 = alpha_[0];
    double a22 = alpha_[1];
    
    double A, B, C;
    double a = a22-a20;
    double temp;
   
    for (size_t i=0;i<nx_*ny_;++i) {
                
        A = a*gxy[i];
        B = a*(gxx[i]-gyy[i]);
        C = -A;

        if (A == 0.0) { // -> linear
            if (B == 0.0) { // -> null, solve
                orientation_[i] = 0.0;
                response_[i] = pointRespM2(i, 0.0);
            } else { // solve linear
                if (C == 0.0) {
                    orientation_[i] = 0.0;
                    response_[i] = pointRespM2(i, 0.0);
                    temp = pointRespM2(i, PI/2.0);
                    if (temp > response_[i]) {
                        response_[i] = temp;
                        orientation_[i] = PI/2.0;
                    }
                } else {
                    orientation_[i] = atan(-C/B);
                    response_[i] = pointRespM2(i, orientation_[i]);
                    temp = pointRespM2(i, orientation_[i]+PI/2.0);
                    if (temp > response_[i]) {
                        response_[i] = temp;
                        orientation_[i] += PI/2.0;
                    }
                }
            }
        } else { // solve quadratic
            double* xRoots = new double[2];
            gsl_poly_solve_quadratic (A, B, C, &xRoots[0], &xRoots[1]);
            
            double* tRoots = new double[2];
            tRoots[0] = atan(xRoots[0]);
            tRoots[1] = atan(xRoots[1]);
            response_[i] = pointRespM2(i, tRoots[0]);
            orientation_[i] = tRoots[0];
            temp = pointRespM2(i, tRoots[1]);
            if (temp > response_[i]) {
                response_[i] = temp;
                orientation_[i] = tRoots[1];
            }
            delete[] xRoots;
            delete[] tRoots;
        }
    }
}


void SteerableDetector::filterM3() {
    double* gx = templates_[0];
    double* gy = templates_[1];
    double* gxxx = templates_[2];
    double* gxxy = templates_[3];
    double* gxyy = templates_[4];
    double* gyyy = templates_[5];
    
    double a10 = alpha_[0];
    double a30 = alpha_[1];
    double a32 = alpha_[2];
    
    double A, B, C, D;
    int nr, nt, deg;
    
    for (size_t i=0;i<nx_*ny_;++i) {
        A = -a10*gx[i] + (2.0*a32-3.0*a30)*gxyy[i] - a32*gxxx[i]; // sin^3
        B =  a10*gy[i] + (3.0*a30-2.0*a32)*gyyy[i] + (7.0*a32-6.0*a30)*gxxy[i]; // sin^2 cos
        C = -a10*gx[i] + (2.0*a32-3.0*a30)*gxxx[i] + (6.0*a30-7.0*a32)*gxyy[i];
        D =  a10*gy[i] + (3.0*a30-2.0*a32)*gxxy[i] + a32*gyyy[i];
        
        A = approxZero(A);
        B = approxZero(B);
        C = approxZero(C);
        D = approxZero(D);
        
        double* roots;
        double* z;
        if (A == 0.0) { // -> quadratic
            if (B == 0.0) { // -> linear
                if (C == 0.0) {// -> null, fixed solution
                    deg = 1;
                    z = new double[2*deg];
                    z[0] = 0.0;
                    z[1] = 0.0;
                } else {
                    deg = 1;
                    double a[2] = {D, C};
                    z = new double[2*deg];
                    gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
                    gsl_poly_complex_solve(a, deg+1, w, z);
                    gsl_poly_complex_workspace_free(w);
                }
            } else { // B!=0
                deg = 2;
                double a[3] = {D, C, B};
                z = new double[2*deg];
                gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
                gsl_poly_complex_solve(a, deg+1, w, z);
                gsl_poly_complex_workspace_free(w);
            }
        } else { // solve cubic
            deg = 3;
            double a[4] = {D, C, B, A};
            z = new double[2*deg];
            gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
            gsl_poly_complex_solve(a, deg+1, w, z);
            gsl_poly_complex_workspace_free(w);
        }
    
        nr = 0;
        for (int k=0;k<deg;++k) {
            if (z[2*k+1]==0.0)
                nr++;
        }
        roots = new double[nr];
        nr = 0;
        for (int k=0;k<deg;++k) {
            if (z[2*k+1]==0.0) {
                roots[nr++] = z[2*k];
            }
        }
        
        double* tRoots;
        if (nr == 0) {
            nt = 4;
            tRoots = new double[nt];
            tRoots[0] = -PI/2.0;
            tRoots[1] = 0.0;
            tRoots[2] = PI/2.0;
            tRoots[3] = PI;
        } else {
            nt = 2*nr;
            tRoots = new double[nt];
            for (int k=0;k<nr;k++) {
                tRoots[k] = atan(roots[k]);
                tRoots[k+nr] = opposite(tRoots[k]);
            }
        }
        delete[] roots;
        
        response_[i] = pointRespM3(i, tRoots[0]);
        orientation_[i] = tRoots[0];
        
        double temp;
        for (int k=1;k<nt;k++) {
            temp = pointRespM3(i, tRoots[k]);
            if (temp > response_[i]) {
                response_[i] = temp;
                orientation_[i] = tRoots[k];
            }
        }
        delete[] tRoots;
        delete[] z;
    }
}


void SteerableDetector::filterM4() {
    double* gxx = templates_[0];
    double* gxy = templates_[1];
    double* gyy = templates_[2];
    double* gxxxx = templates_[3];
    double* gxxxy = templates_[4];
    double* gxxyy = templates_[5];
    double* gxyyy = templates_[6];
    double* gyyyy = templates_[7];

    double a20 = alpha_[0];
    double a22 = alpha_[1];
    double a40 = alpha_[2];
    double a42 = alpha_[3];
    double a44 = alpha_[4];
    
    double A, B, C, D, E;
    
    int nr, deg;
    double delta;
    
    for (size_t i=0;i<nx_*ny_;i++) {
        
        A = (a22-a20)*gxy[i] + (a42-2.0*a40)*gxyyy[i] + (2.0*a44-a42)*gxxxy[i];
        B = (a20-a22)*gyy[i] + (a22-a20)*gxx[i] + (2.0*a40-a42)*gyyyy[i] + 6.0*(a42-a40-a44)*gxxyy[i] + (2.0*a44-a42)*gxxxx[i];
        C = 6.0*((a40-a42+ a44)*gxyyy[i] + (a42-a40-a44)*gxxxy[i]);
        D = (a20-a22)*gyy[i] + (a22-a20)*gxx[i] + (a42-2.0*a44)*gyyyy[i] + 6.0*(a40-a42+a44)*gxxyy[i] + (a42-2.0*a40)*gxxxx[i];
        E = (a20-a22)*gxy[i] + (a42-2.0*a44)*gxyyy[i] + (2.0*a40-a42)*gxxxy[i];
        
        A = approxZero(A);
        C = approxZero(C);
        E = approxZero(E);
        
        double* roots;
        double* z;
        if (A == 0.0) { // -> cubic
            if (B == 0.0) { // -> quadratic
                if (C == 0.0) { // -> linear
                    if (D == 0.0) { // solve null
                        deg = 1;
                        z = new double[2*deg];
                        z[0] = 0.0;
                        z[1] = 0.0;
                    } else { // solve linear
                        deg = 1;
                        double a[2] = {E, D};
                        z = new double[2*deg];
                        gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
                        gsl_poly_complex_solve(a, deg+1, w, z);
                        gsl_poly_complex_workspace_free(w);
                    }
                } else { // solve quadratic
                    deg = 2;
                    double a[3] = {E, D, C};
                    z = new double[2*deg];
                    gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
                    gsl_poly_complex_solve(a, deg+1, w, z);
                    gsl_poly_complex_workspace_free(w);
                }
            } else { // solve cubic
                if ( (C == 0.0) && (E == 0.0) ) {
                    delta = -D/B;
                    if (delta > 0.0) {
                        deg = 3;
                        delta = sqrt(delta);
                        z = new double[2*deg];
                        z[0] = 0.0;
                        z[1] = 0.0;
                        z[2] = delta;
                        z[3] = 0.0;
                        z[4] = -delta;
                        z[5] = 0.0;
                    } else {
                        deg = 1;
                        z = new double[2*deg];
                        z[0] = 0.0;
                        z[1] = 0.0;
                    }
                } else {
                    deg = 3;
                    double a[4] = {E, D, C, B};
                    z = new double[2*deg];
                    gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
                    gsl_poly_complex_solve(a, deg+1, w, z);
                    gsl_poly_complex_workspace_free(w);
                }
            }
        } else { // solve quartic
            deg = 4;
            double a[5] = {E, D, C, B, A};
            z = new double[2*deg];
            gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
            gsl_poly_complex_solve(a, deg+1, w, z);
            gsl_poly_complex_workspace_free(w);
        }
            
        // # real roots
        nr = 0;
        for (int k=0;k<deg;++k) {
            if (z[2*k+1]==0.0)
                nr++;
        }
        roots = new double[nr];
        nr = 0;
        for (int k=0;k<deg;++k) {
            if (z[2*k+1]==0.0) {
                roots[nr++] = z[2*k];
            }
        }
        
        double* tRoots;
        if (nr == 0) { //orientation[i] = 0.0;
            nr = 2;
            tRoots = new double[nr];
            tRoots[0] = 0.0;
            tRoots[1] = PI/2.0;
        } else {
            if (roots[0] == 0.0) {
                nr++;
                tRoots = new double[nr];
                for (int k=0;k<nr-1;k++) {
                    tRoots[k] = atan(roots[k]);
                }
                tRoots[nr-1] = PI/2;
            } else {
                tRoots = new double[nr];
                for (int k=0;k<nr;k++) {
                    tRoots[k] = atan(roots[k]);
                }
            }
        }
        delete[] roots;
              
        response_[i] = pointRespM4(i, tRoots[0]);
        orientation_[i] = tRoots[0];
       
        double temp;
        for (int k=1;k<nr;k++) {
            temp = pointRespM4(i, tRoots[k]);
            if (temp > response_[i]) {
                response_[i] = temp;
                orientation_[i] = tRoots[k];
            }
        }
        delete[] z;
        delete[] tRoots;
    }
}


void SteerableDetector::filterM5() {
    double* gx = templates_[0];
    double* gy = templates_[1];
    double* gxxx = templates_[2];
    double* gxxy = templates_[3];
    double* gxyy = templates_[4];
    double* gyyy = templates_[5];
    double* gxxxxx = templates_[6];
    double* gxxxxy = templates_[7];
    double* gxxxyy = templates_[8];
    double* gxxyyy = templates_[9];
    double* gxyyyy = templates_[10];
    double* gyyyyy = templates_[11];
    
    double a10 = alpha_[0];
    double a30 = alpha_[1];
    double a32 = alpha_[2];
    double a52 = alpha_[3];
    double a54 = alpha_[4];
    
    double A, B, C, D, E, F;
    int nr, nt, deg;
    double delta;
    
    for (size_t i=0;i<nx_*ny_;++i) {

        A = -a10*gx[i] + (2.0*a32-3.0*a30)*gxyy[i] - a32*gxxx[i] + (4.0*a54-3.0*a52)*gxxxyy[i] + 2.0*a52*gxyyyy[i] -a54*gxxxxx[i];
        B = a10*gy[i] + (3.0*a30-2.0*a32)*gyyy[i] + (7.0*a32-6.0*a30)*gxxy[i] - 2.0*a52*gyyyyy[i] + (17.0*a52-12.0*a54)*gxxyyy[i] + (13.0*a54-6.0*a52)*gxxxxy[i];
        C = -2.0*a10*gx[i] + (3.0*a30-5.0*a32)*gxyy[i] + (a32-3.0*a30)*gxxx[i] + (12.0*a54-17.0*a52)*gxyyyy[i] + (30.0*a52-34.0*a54)*gxxxyy[i] + (4.0*a54-3.0*a52)*gxxxxx[i];
        D = 2.0*a10*gy[i] + (5.0*a32-3.0*a30)*gxxy[i] + (3.0*a30-a32)*gyyy[i] + (17.0*a52-12.0*a54)*gxxxxy[i] + (34.0*a54-30.0*a52)*gxxyyy[i] + (3.0*a52-4.0*a54)*gyyyyy[i];
        E = -a10*gx[i] + (2.0*a32-3.0*a30)*gxxx[i] + (6.0*a30-7.0*a32)*gxyy[i] + 2.0*a52*gxxxxx[i] + (12.0*a54-17.0*a52)*gxxxyy[i] + (6.0*a52-13.0*a54)*gxyyyy[i];
        F = a10*gy[i] + (3.0*a30-2.0*a32)*gxxy[i] + a32*gyyy[i] + (3.0*a52-4.0*a54)*gxxyyy[i] - 2.0*a52*gxxxxy[i]+ a54*gyyyyy[i];
        
        A = approxZero(A);
        B = approxZero(B);
        C = approxZero(C);
        D = approxZero(D);
        E = approxZero(E);
        F = approxZero(F);
        
        double* roots;
        double* z;
        
        if (A == 0.0) { // quartic
            if (B == 0.0) { // cubic
                if (C == 0.0) { // quadratic
                    if (D == 0.0) { // linear
                        if (E == 0.0) { // null
                            deg = 1;
                            z = new double[2*deg];
                            z[0] = 0.0;
                            z[1] = 0.0;
                        } else {
                            deg = 1;
                            double a[2] = {E, D};
                            z = new double[2*deg];
                            gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
                            gsl_poly_complex_solve(a, deg+1, w, z);
                            gsl_poly_complex_workspace_free(w);
                        }
                    } else { // solve quadratic
                        deg = 2;
                        double a[3] = {E, D, C};
                        z = new double[2*deg];
                        gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
                        gsl_poly_complex_solve(a, deg+1, w, z);
                        gsl_poly_complex_workspace_free(w);
                    }
                } else { // solve cubic
                    if ( (D == 0.0) && (F == 0.0) ) {
                        delta = -E/C;
                        if (delta > 0.0) {
                            deg = 3;
                            delta = sqrt(delta);
                            z = new double[2*deg];
                            z[0] = 0.0;
                            z[1] = 0.0;
                            z[2] = delta;
                            z[3] = 0.0;
                            z[4] = -delta;
                            z[5] = 0.0;
                        } else {
                            deg = 1;
                            z = new double[2*deg];
                            z[0] = 0.0;
                            z[1] = 0.0;
                        }
                    } else {
                        deg = 3;
                        double a[4] = {F, E, D, C};
                        z = new double[2*deg];
                        gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
                        gsl_poly_complex_solve(a, deg+1, w, z);
                        gsl_poly_complex_workspace_free(w);
                    }
                }
            } else { // solve quartic
                deg = 4;
                double a[5] = {F, E, D, C, B};
                z = new double[2*deg];
                gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
                gsl_poly_complex_solve(a, deg+1, w, z);
                gsl_poly_complex_workspace_free(w);
            }
        } else {
            deg = 5;
            double a[6] = {F, E, D, C, B, A};
            z = new double[2*deg];
            gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc(deg+1);
            gsl_poly_complex_solve(a, deg+1, w, z);
            gsl_poly_complex_workspace_free(w);
        }

        
        // # real roots
        nr = 0;
        for (int k=0;k<deg;++k) {
            if (z[2*k+1]==0.0)
                nr++;
        }
        roots = new double[nr];
        nr = 0;
        for (int k=0;k<deg;++k) {
            if (z[2*k+1]==0.0) {
                roots[nr++] = z[2*k];
            }
        }
        
        double* tRoots;
        if (nr == 0) { //orientation[i] = 0.0;
            nt = 4;
            tRoots = new double[nt];
            tRoots[0] = -PI/2.0;
            tRoots[1] = 0.0;
            tRoots[2] = PI/2.0;
            tRoots[3] = PI;
        } else {
            nt = 2*nr;
            tRoots = new double[nt];
            for (int k=0;k<nr;k++) {
                tRoots[k] = atan(roots[k]);
                tRoots[k+nr] = opposite(tRoots[k]);
            }
        }
        delete[] roots;
              
        response_[i] = pointRespM5(i, tRoots[0]);
        orientation_[i] = tRoots[0];
       
        double temp;
        for (int k=1;k<nt;k++) {
            temp = pointRespM5(i, tRoots[k]);
            if (temp > response_[i]) {
                response_[i] = temp;
                orientation_[i] = tRoots[k];
            }
        }
        delete[] z;
        delete[] tRoots;
    }
}


int SteerableDetector::getNumTemplates(const int M) {
    if (M==1) {
        return M*(M+3)/2;
    } else {
        return M*(M+3)/2 - getNumTemplates(M-1);
    }
}


double SteerableDetector::approxZero(const double n) {
    if (fabs(n) < TOL) {
        return 0.0;
    } else {
        return n;
    }
}


double SteerableDetector::opposite(const double theta) {
    if (theta > 0.0) { // values in (-PI,PI]
        return theta - PI;
    } else {
        return theta + PI;
    }
}


int SteerableDetector::mirror(const int x, const int nx) {
    // mirror position in image domain for interpolation
    if (x >= 0 && x < nx) {
        return x;
    } else if (x < 0) {
        return -x;
    } else {
        return 2*nx-2-x;
    }
}


double SteerableDetector::interp(const double* image, const int nx, const int ny, const double x, const double y) {
    // linear interpolation
    int xi = (int)x;
    int yi = (int)y;
    int x0, x1, y0, y1;
    
    double dx = x-xi;
    double dy = y-yi;
    if (x < 0) { dx = -dx; x1 = mirror(xi-1, nx); } else { x1 = mirror(xi+1, nx); }
    if (y < 0) { dy = -dy; y1 = mirror(yi-1, ny); } else { y1 = mirror(yi+1, ny); }
    x0 = mirror(xi, nx);
    y0 = mirror(yi, ny);
    return (1.0-dy)*(dx*image[x1+y0*nx] + (1.0-dx)*image[x0+y0*nx]) + dy*(dx*image[x1+y1*nx] + (1.0-dx)*image[x0+y1*nx]);
}

}  // namespace steerable
