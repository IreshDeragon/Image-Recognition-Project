#ifndef UNTITLED_PMC_H
#define UNTITLED_PMC_H

#include <string>
#include <iostream>

#define STRING(num) STR(num)
#define STR(num) #num

#include <cstdint>

class PMC {

public:
    float ***weights;
    int *nbNeurons;
    int nbEntry;
    int nbOut;
    int layer;
    float **inputs;

    PMC(int32_t layer, int32_t* nbNeurons, int32_t nbEntry, int32_t nbOut);

    float calculTotalPredict(int32_t layer, int32_t nbInput, int32_t output, float *input);

    float calculTotalSigma(int32_t layer, int32_t nbInput, int32_t output, float *sigma);

    float *predict(float *entry);

    void train(int32_t epoch, float LR, float **points, float **Y, int32_t pointsSize);

    void updateWeights(int32_t layer, int32_t nbInput, int32_t output, float *inputs, float sigma, float LR);

    void tostring();

    void calcul_sigma(float **sigma);

    ~PMC();
};


#endif
