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

#include "mex.h"
#include "steerableDetector.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // check # inputs
    if (nrhs < 3 || nrhs > 5)
        mexErrMsgTxt("Required inputs arguments: image, filter order, sigma.\nOptional: # angles for rotations output; border condition (see help).");
    if (nlhs > 4)
        mexErrMsgTxt("Too many output arguments.");

    // check image
    if (!mxIsDouble(prhs[0]) || mxGetNumberOfDimensions(prhs[0]) != 2)
        mexErrMsgTxt("Input image must be a 2-D array.");
    size_t nx = (size_t)mxGetN(prhs[0]); // cols
    size_t ny = (size_t)mxGetM(prhs[0]);
    int N = nx*ny;
    double* input = mxGetPr(prhs[0]);
    // check for NaNs in input, as these will result in a crash
    for (int i=0;i<N;++i) {
        if (mxIsNaN(input[i])) {
            mexErrMsgTxt("Input image contains NaNs.");
            break;
        }
    }

    // check order
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1 || *mxGetPr(prhs[1])<1 || *mxGetPr(prhs[1])>5)
        mexErrMsgTxt("The order 'M' must be an integer between 1 and 5.");
    int M = (int) *mxGetPr(prhs[1]);

    // check sigma
    if (!mxIsDouble(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1 || *mxGetPr(prhs[2]) <= 0.0)
        mexErrMsgTxt("Sigma must be a strictly positive scalar value.");
    double sigma = *mxGetPr(prhs[2]);

    // Set defaults for options
    int borderCondition = 3;
    size_t nt = 36;

    // check 1st option: angles or border condition
    if (nrhs >= 4) {
        for (int i=3;i<nrhs;++i) {

            if (mxIsDouble(prhs[i]) && *mxGetPr(prhs[i])>0) { // # angles
                nt = (size_t) *mxGetPr(prhs[i]);
            } else if (mxIsChar(prhs[i])) { // border condition
                size_t nchar = mxGetNumberOfElements(prhs[i])+1;
                char *ch = new char[nchar];
                int f = mxGetString(prhs[i], ch,  nchar);
                if (f!=0) {
                    mexErrMsgTxt("Error parsing border condition.");
                }
                std::string str = ch;
                delete[] ch;
                int (*pf)(int) = tolower;
                transform(str.begin(), str.end(), str.begin(), pf);
                if (str.compare("mirror")==0) {
                    borderCondition = 3;
                } else if (str.compare("periodic")==0) {
                    borderCondition = 2;
                } else if (str.compare("replicate")==0) {
                    borderCondition = 1;
                } else if (str.compare("zeros")==0) {
                    borderCondition = 0;
                } else {
                    mexErrMsgTxt("Unsupported border conditions.");
                }
            } else {
                mexErrMsgTxt("Allowed options: # angles (must be ? 1) or border condition.");
            }
        }

    }

    int L = 2*(int)(4.0*sigma)+1; // support of the Gaussian kernels

    if (L>nx || L>ny) {
        mexPrintf("Sigma must be smaller than %.2f\n", (std::min(nx,ny)-1)/8.0);
        mexErrMsgTxt("Sigma value results in filter support that is larger than image.");
    }

    double* pixels = new double[N];

    // Process inputs
    // Switch matrix to row-major (Matlab uses column-major)
    div_t divRes;
    for (int i=0;i<N;++i) {
        divRes = div(i, ny);
        pixels[divRes.quot+divRes.rem*nx] = input[i];
    }

    steerable::SteerableDetector sd = steerable::SteerableDetector(pixels, nx, ny, M, sigma, borderCondition);
    sd.filter();

    // Process outputs
    // Switch outputs back to column-major format
    if (nlhs > 0) {
        for (int i=0;i<N;++i) {
            divRes = div(i, nx);
            pixels[divRes.quot+divRes.rem*ny] = sd.response_[i];
        }
        plhs[0] = mxCreateDoubleMatrix(ny, nx, mxREAL);
        memcpy(mxGetPr(plhs[0]), pixels, N*sizeof(double));
    }

    if (nlhs > 1) { // return orientation map
        for (int i=0;i<N;++i) {
            divRes = div(i, nx);
            pixels[divRes.quot+divRes.rem*ny] = sd.orientation_[i];
        }
        plhs[1] = mxCreateDoubleMatrix(ny, nx, mxREAL);
        memcpy(mxGetPr(plhs[1]), pixels, N*sizeof(double));
    }

    if (nlhs > 2) { // run NMS
        sd.runNMS();
        for (int i=0;i<N;++i) {
            divRes = div(i, nx);
            pixels[divRes.quot+divRes.rem*ny] = sd.nms_response_[i];
        }
        plhs[2] = mxCreateDoubleMatrix(ny, nx, mxREAL);
        memcpy(mxGetPr(plhs[2]), pixels, N*sizeof(double));
    }

    if (nlhs > 3) { // return filterbank
        const mwSize dims[] = {ny, nx, nt};
        plhs[3] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        double* p = mxGetPr(plhs[3]);
        sd.getAngleResponse(p, nt);
    }

    // Free memory
    delete[] pixels;
}
