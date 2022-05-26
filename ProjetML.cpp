#include "ProjetML.h"
#include <stdlib.h>
#include <string>
#include <iostream>
#include <math.h>

Model createModelPMC(int layer, int *nbNeurons, int nbEntry, int nbOut) {
    struct Model model;
    model.layer = layer;
    model.nbNeurons = nbNeurons;
    model.nbEntry = nbEntry;
    model.nbOut = nbOut;

    //Instantiation du tableau de poids
    if(layer!=0){
        model.weights = new double **[layer + 1];
        //couche 0
        model.weights[0] = new double *[nbNeurons[0]];
        for (int j = 0; j < nbNeurons[0]; j++) {
            model.weights[0][j] = new double[nbEntry + 1];
            for (int w = 0; w < nbEntry + 1; w++) {
                model.weights[0][j][w] = (double) rand() / RAND_MAX * 2 - 1;
            }
        }
        //autres couches
        for (int i = 0; i < layer; i++) {
            model.weights[i] = new double *[nbNeurons[i]];
            for (int j = 0; j < nbNeurons[i]; j++) {
                model.weights[i][j] = new double[nbNeurons[i - 1] + 1];
                for (int w = 0; w < nbNeurons[i - 1] + 1; w++) {
                    model.weights[i][j][w] = (double) rand() / RAND_MAX * 2 - 1;
                }
            }
        }
        //dernière couche
        model.weights[layer] = new double *[nbOut];
        for (int i = 0; i < nbOut; i++) {
            model.weights[layer][i] = new double[nbNeurons[layer - 1] + 1];
            for (int j = 0; j < nbNeurons[layer - 1] + 1; j++) {
                model.weights[layer][i][j] = (double) rand() / RAND_MAX * 2 - 1;
            }
        }
    }else{ //si layer est égal à 0
        model.weights = new double **[1];
        //couche 0
        model.weights[0] = new double *[nbOut];
        for (int j = 0; j < nbOut; j++) {
            model.weights[0][j] = new double[nbEntry + 1];
            for (int w = 0; w < nbEntry + 1; w++) {
                model.weights[0][j][w] = (double) rand() / RAND_MAX * 2 - 1;
            }
        }
    }



    //Instatiation du tableau d'input (spikes)
    model.inputs = new double *[layer + 1];
    for (int i = 0; i < layer + 1; i++) {
        if (i == 0) {
            model.inputs[i] = new double[nbEntry + 1];
        } else if (i == layer) {
            model.inputs[i] = new double[nbOut + 1];
        } else {
            model.inputs[i] = new double[nbNeurons[i] + 1];
        }
    }
    return model;
}

/**
 *
 * Fonctions permettant d'implémenter le Predict
 *
 **/

double calculTotalPredict(Model model, int layer, int nbInput, int output, double *input) {
    double total = model.weights[layer][output][0];
    for (int t = 0; t < nbInput; t++) {
        total += model.weights[layer][output][t+1] * input[t];
    }
    return tanh(total);
}

double *predictClassPMC(Model model, double *entry) {
    double total;
    //calcul résultats des neurones 1ère couche
    for (int j = 0; j < model.nbNeurons[0]; j++) {
        model.inputs[0][j] = calculTotalPredict(model, 0, model.nbEntry, j, entry);
    }
    //calcul résultats autres neurones
    for (int i = 1; i < model.layer; i++) {
        for (int j = 0; j < model.nbNeurons[i]; j++) {
            model.inputs[i][j] = calculTotalPredict(model, i, model.nbNeurons[i - 1], j, model.inputs[i - 1]);
        }
    }
    //calcul résultats dernière couche
    for (int i = 0; i < model.nbOut; i++) {
        model.inputs[model.layer][i] = calculTotalPredict(model, model.layer, model.nbNeurons[model.layer - 1], i, model.inputs[model.layer - 1]);
    }

    return model.inputs[model.layer];
}


/**
 *
 * Fonctions permettant d'implémenter le Train
 *
 **/
double calculTotalSigma(Model model, int layer, int nbOutput, int input, double *sigma) {
    double total = 0;
    for (int output = 0; output < nbOutput; output++) {
        total += model.weights[layer][output][input] * sigma[output];
    }
    return total;
}

void calcul_sigma(Model model, double **sigma) {
    //calcul
    double total = 0;
    for (int i = 0; i < model.nbNeurons[model.layer - 1]; i++) {
        total = calculTotalSigma(model, model.layer, model.nbOut, i, sigma[model.layer]); //total avant dernière couche
        sigma[model.layer - 1][i] = (1.0 - (model.inputs[model.layer - 1][i] * model.inputs[model.layer - 1][i])) * total;
    }

    for (int l = model.layer - 1; l > 0; l--) {
        for (int i = 0; i < model.nbNeurons[l - 1]; i++) {
            total = calculTotalSigma(model, l, model.nbNeurons[l], i, sigma[l]);
            sigma[l - 1][i] = (1.0 - (model.inputs[l - 1][i] * model.inputs[l - 1][i])) * total; //total couches précédentes
        }
    }
}

void trainClassPMC(Model model, int epochs, double LR, double **points, double **Y, int pointsSize, int YSize) {
    //initialisation sigma[]
    double **sigma = new double *[model.layer + 1];
    for (int i = 0; i < model.layer + 1; i++) {
        if (i == model.layer) {
            sigma[i] = new double[model.nbOut];
        } else {
            sigma[i] = new double[model.nbNeurons[i] + 1];
        }
    }

    //Itérations sur le batch
    for (int epoch = 0; epoch < epochs; epoch++) {
        int point = rand() % pointsSize;
        double *result = predictClassPMC(model, points[point]);


        //calcul sigma dernière couche
        for (int i = 0; i < model.nbOut; i++) {
            sigma[model.layer][i] = (1 - (result[i] * result[i])) * (result[i] - Y[point][i]);
        }
        //calcul des autres sigma
        calcul_sigma(model, sigma);


        //mise à jour des poids
        //couche 0
        for (int output = 0; output < model.nbNeurons[0]; output++) {
            model.weights[0][output][0] -= LR * sigma[0][output];
            updateWeights(model, 0, model.nbEntry, output, points[point], sigma[0][output], LR);
        }
        //autres couches
        for (int l = 1; l < model.layer; l++) {
            for (int output = 0; output < model.nbNeurons[l]; output++) {
                model.weights[l][output][0] -= LR * sigma[l][output];
                updateWeights(model, l, model.nbNeurons[l - 1], output, model.inputs[l - 1], sigma[l][output], LR);
            }
        }
        //dernière couche
        for (int output = 0; output < model.nbOut; output++) {
            model.weights[model.layer][output][0] -= LR * sigma[model.layer][output];
            updateWeights(model, model.layer, model.nbNeurons[model.layer - 1], output, model.inputs[model.layer - 1], sigma[model.layer][output], LR);
        }

    }
}

void updateWeights(Model model, int layer, int nbInput, int output, double *inputs, double sigma, double LR) {
    for (int input = 1; input < nbInput + 1; input++) {
        model.weights[layer][output][input] -= LR * inputs[input - 1] * sigma;
    }
}


/**
 *
 * ToString
 *
 **/

void tostring(Model model) {
    std::string result = "";
    for (int i = 0; i < model.layer + 1; i++) {
        if (i != model.layer) {
            for (int j = 0; j < model.nbNeurons[i]; j++) {
                if (i == 0) {
                    result += "Neuron: ";
                    for (int w = 0; w < model.nbEntry + 1; w++) {
                        result += " ";
                        result += std::to_string(model.weights[i][j][w]);
                    }

                } else {
                    result += "Neuron: ";
                    for (int w = 0; w < model.nbNeurons[i - 1] + 1; w++) {
                        result += " ";
                        result += std::to_string(model.weights[i][j][w]);
                        //result += " : ";
                        //result += std::to_string(this->inputs[i][j][w]);
                    }

                }
                result += " [";
                result += std::to_string(model.inputs[i][j]);
                result += "] ";

                result += "      ";
            }
        } else {
            for (int j = 0; j < model.nbOut; j++) {
                for (int w = 0; w < model.nbNeurons[i - 1] + 1; w++) {
                    result += " ";
                    result += std::to_string(model.weights[i][j][w]);
                }
                result += " [";
                result += std::to_string(model.inputs[i][j]);
                result += "] ";
            }
        }

        result += "\n\n";
    }
    std::cout << result << std::endl;
}