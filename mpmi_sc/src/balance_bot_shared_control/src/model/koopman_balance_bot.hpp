#ifndef KOOPMAN_OPERATOR_BALANCE_BOT_HPP
#define KOOPMAN_OPERATOR_BALANCE_BOT_HPP

#include <math.h>
#include <fstream>
#include <sstream>
#include "basis_balance_bot.hpp"

#define IDX2C(i ,j, ld) (((j)*(ld))+(i))
#define KOOPMAN_DIM 6

class KoopmanOperator
{

public:
    std::vector<float> _K_cpu;
    float* _K_gpu = new float[KOOPMAN_DIM*KOOPMAN_DIM]();
    Basis* basis;

    KoopmanOperator(Basis* _basis) {
      basis = _basis;
    }

    ~KoopmanOperator() {
      delete [] _K_gpu;
    }

    void loadOperator(std::string filePath) {

      // load koopman from csv in row-major order
      std::ifstream file(filePath);
      std::string line;
      while(getline(file, line)){
        std::stringstream ss(line);
        while(getline(ss, line, ',')){
          _K_cpu.push_back(std::stof(line));
        }
      }
      file.close();

      // cuda will expect koopman in column-major order
      int counter = 0;
      for (int i = 0; i < KOOPMAN_DIM; i++) {
        for (int j = 0; j < KOOPMAN_DIM; j++) {
          _K_gpu[counter] = _K_cpu[IDX2C(i,j,KOOPMAN_DIM)];
          counter++;
        }
      }

      std::cout << "Loaded Koopman Operator." << std::endl;
    }

};

#endif
