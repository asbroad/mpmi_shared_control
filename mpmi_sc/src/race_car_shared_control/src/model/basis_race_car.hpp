#ifndef BASIS_RACE_CAR_HPP
#define BASIS_RACE_CAR_HPP

#include <math.h>

#define KOOPMAN_DIM 36

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
            x[4],
            x[5], // state variables
            x[6],
            x[7],
            x[8], // control variables
            x[6]*x[0],
            x[6]*x[1],
            x[6]*x[3],
            x[6]*x[4],
            x[7]*x[0], // control * state
            x[7]*x[1],
            x[7]*x[3],
            x[7]*x[4],
            x[8]*x[0],
            x[8]*x[3],
            x[8]*x[4],
            x[0]*(float)cos(x[2]), // heading * control
            x[0]*(float)sin(x[2]), // heading * state
            x[1]*(float)sin(x[2]),
            x[3]*(float)cos(x[2]),
            x[4]*(float)cos(x[2]),
            x[3]*x[0],
            x[4]*x[1],
            x[5]*x[2], // velocity * position
            x[0]*(float)cos(x[5]),
            x[0]*(float)sin(x[5]), // heading velocity * state
            x[1]*(float)cos(x[5]),
            x[1]*(float)sin(x[5]),
            x[3]*(float)cos(x[5]),
            x[3]*(float)sin(x[5]),
            x[4]*(float)sin(x[5]),
      };
      return projection;
    }

};

#endif
