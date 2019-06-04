#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "mat.h"

using namespace std;

int main()
{

    Matrix trainingData;
    Matrix a;

    initRand();

    trainingData.read();

    a = trainingData.normalizeCols();
    //trainingData.write();
    Matrix weights = trainingData;
    weights.rand(0.00, 1.00);
    weights.write();

    return 0;
}