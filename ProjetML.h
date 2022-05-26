//
// Created by ALASMI on 26/05/2022.
//

#ifndef UNTITLED_PROJETML_H
#define UNTITLED_PROJETML_H

struct Model {
    double ***weights;
    int *nbNeurons;
    int nbEntry;
    int nbOut;
    int layer;
    double **inputs;
};


Model createModelPMC(int layer, int *nbNeurons, int nbEntry, int nbOut);

double calculTotalPredict(Model model, int layer, int nbInput, int output, double *input);

double calculTotalSigma(Model model, int layer, int nbInput, int output, double *sigma);

double *predictClassPMC(Model model, double *entry);

void trainClassPMC(Model model, int epoch, double LR, double **points, double **Y, int pointsSize, int YSize);

void updateWeights(Model model, int layer, int nbInput, int output, double *inputs, double sigma, double LR);

void tostring(Model model);

void calcul_sigma(Model model, double **sigma);


#endif //UNTITLED_PROJETML_H
