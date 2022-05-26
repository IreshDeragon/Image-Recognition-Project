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
    double* predict(double* entry);
    void train(int epoch, double LR, double** points, double* Y, int pointsSize, int YSize);
    void tostring();

};



#endif
