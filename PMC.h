#ifndef UNDEFINED_PMC
#define UNDEFINED_PMC

#include <string>
#include <iostream>
#include <stdlib.h>

#define STRING(num) STR(num)

#include <cstdint>

class PMC {

public:
    float ***weights;
    int *nbNeurons;
    int nbEntry;
    int nbOut;
    int layer;
    float **inputs;

    PMC(int layer, int *nbNeurons, int nbEntry, int nbOut);

    float calculTotalPredict(int layer, int nbInput, int output, float *input);

    float calculTotalSigma(int layer, int nbInput, int output, float *sigma);

    float *predict(float *entry);

    void train(int epoch, float LR, float **points, float **Y, int pointsSize);

    void updateWeights(int layer, int nbInput, int output, float *inputs, float sigma, float LR);

    void tostring();

    void calcul_sigma(float **sigma);

    ~PMC();
};


#endif
