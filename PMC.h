#ifndef UNTITLED_PMC_H
#define UNTITLED_PMC_H
#include <string>
#include <iostream>
#define STRING(num) STR(num)
#define STR(num) #num


class PMC {

    public:
    double*** weights;
    int* nbNeurons;
    int nbEntry;
    int nbOut;
    int layer;
    double** inputs;
    PMC(int layer, int* nbNeurons, int nbEntry, int nbOut);
    double calculTotalPredict(int layer, int nbInput, int output, double *input);
    double calculTotalSigma(int layer, int nbInput, int output, double *sigma);
    double* predict(double* entry);
    void train(int epoch, double LR, double** points, double** Y, int pointsSize, int YSize);
    void updateWeights(int layer, int nbInput, int output, double* inputs, double sigma, double LR);
    void tostring();
    void calcul_sigma(double** sigma);

};



#endif
