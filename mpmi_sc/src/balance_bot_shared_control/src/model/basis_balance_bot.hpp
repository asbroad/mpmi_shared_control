#ifndef BASIS_BALANCE_BOT_HPP
#define BASIS_BALANCE_BOT_HPP

#include <math.h>

#define KOOPMAN_DIM 6

class Basis {

public:

    Basis() {
    }

    ~Basis() {
    }

    float* project(const float* x) {
      float* projection = new float[KOOPMAN_DIM]{
            1,
            x[0],
            x[1],
            x[2],
            x[3],
            x[2]*(float)cos(x[0])
      };
      return projection;
    }

};

#endif
