## Steerable filters
<img style="float: right;" src="http://www.francoisaguet.net/img/dnaHSV.jpg">


Steerable filters for edge and ridge detection from [Jacob and Unser, 2004](http://ieeexplore.ieee.org/document/1307008) implemented in C++ with wrappers for Python and Matlab.

Requirements:
* [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/) must be installed.
  
  On Ubuntu/Debian:
  ```bash
  sudo apt install libgsl-dev
  ```

### Python

Installation:
```bash
pip3 install -e .
```
See [example.py](python/example.py) for usage instructions.

### Matlab

Installation:
1. Add `steerable/matlab` to the Matlab path.
2. Compile the MEX file.
```matlab
cd matlab
mex -I/usr/local/include -I../src /usr/local/lib/libgsl.a -output steerableDetector ../src/steerableDetector.cpp ../src/steerableDetector_mex.cpp
```

Documentation:
```matlab
help steerableDetector
```
