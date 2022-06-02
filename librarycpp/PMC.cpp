#include "PMC.h"
#include <stdlib.h>
#include <string>
#include <iostream>
#include <math.h>


PMC::PMC(int layer, int *nbNeurons, int nbEntry, int nbOut) {
    this->layer = layer;
    this->nbNeurons = nbNeurons;
    this->nbEntry = nbEntry;
    this->nbOut = nbOut;

    //Instantiation du tableau de poids
    if(layer!=0){
        weights = new double **[layer + 1];
        //couche 0
        weights[0] = new double *[nbNeurons[0]];
        for (int j = 0; j < nbNeurons[0]; j++) {
            weights[0][j] = new double[nbEntry + 1];
            for (int w = 0; w < nbEntry + 1; w++) {
                weights[0][j][w] = (double) rand() / RAND_MAX * 2 - 1;
            }
        }
        //autres couches
        for (int i = 1; i < layer; i++) {
            weights[i] = new double *[nbNeurons[i]];
            for (int j = 0; j < nbNeurons[i]; j++) {
                weights[i][j] = new double[nbNeurons[i - 1] + 1];
                for (int w = 0; w < nbNeurons[i - 1] + 1; w++) {
                    weights[i][j][w] = (double) rand() / RAND_MAX * 2 - 1;
                }
            }
        }
        //dernière couche
        weights[layer] = new double *[nbOut];
        for (int i = 0; i < nbOut; i++) {
            weights[layer][i] = new double[nbNeurons[layer - 1] + 1];
            for (int j = 0; j < nbNeurons[layer - 1] + 1; j++) {
                weights[layer][i][j] = (double) rand() / RAND_MAX * 2 - 1;
            }
        }
    }else{ //si layer est égal à 0
        weights = new double **[1];
        //couche 0
        weights[0] = new double *[nbOut];
        for (int j = 0; j < nbOut; j++) {
            weights[0][j] = new double[nbEntry + 1];
            for (int w = 0; w < nbEntry + 1; w++) {
                weights[0][j][w] = (double) rand() / RAND_MAX * 2 - 1;
            }
        }
    }



    //Instatiation du tableau d'input (spikes)
    inputs = new double *[layer + 1];
    for (int i = 0; i < layer + 1; i++) {
        if (i == 0) {
            inputs[i] = new double[nbEntry + 1];
        } else if (i == layer) {
            inputs[i] = new double[nbOut + 1];
        } else {
            inputs[i] = new double[nbNeurons[i] + 1];
        }
    }
}

/**
 *
 * Fonctions permettant d'implémenter le Predict
 *
 **/

double PMC::calculTotalPredict(int layer, int nbInput, int output, double *input) {
    double total = this->weights[layer][output][0];
    for (int t = 0; t < nbInput; t++) {
        total += this->weights[layer][output][t+1] * input[t];
    }
    return tanh(total);
}

double *PMC::predict(double *entry) {
    double total;
    //calcul résultats des neurones 1ère couche
    for (int j = 0; j < this->nbNeurons[0]; j++) {
        this->inputs[0][j] = calculTotalPredict(0, nbEntry, j, entry);
    }
    //calcul résultats autres neurones
    for (int i = 1; i < layer; i++) {
        for (int j = 0; j < this->nbNeurons[i]; j++) {
            this->inputs[i][j] = calculTotalPredict(i, nbNeurons[i - 1], j, this->inputs[i - 1]);
        }
    }
    //calcul résultats dernière couche
    for (int i = 0; i < this->nbOut; i++) {
        this->inputs[layer][i] = calculTotalPredict(layer, nbNeurons[layer - 1], i, this->inputs[layer - 1]);
    }

    return this->inputs[layer];
}
double PMC::calculTotalPredictReg(int layer, int nbInput, int output, double *input) {
    double total = this->weights[layer][output][0];
    for (int t = 0; t < nbInput; t++) {
        total += this->weights[layer][output][t+1] * input[t];
    }
    return total;
}

double *PMC::predictReg(double *entry) {
    double total;
    //calcul résultats des neurones 1ère couche
    for (int j = 0; j < this->nbNeurons[0]; j++) {
        this->inputs[0][j] = calculTotalPredict(0, nbEntry, j, entry);
    }
    //calcul résultats autres neurones
    for (int i = 1; i < layer; i++) {
        for (int j = 0; j < this->nbNeurons[i]; j++) {
            this->inputs[i][j] = calculTotalPredict(i, nbNeurons[i - 1], j, this->inputs[i - 1]);
        }
    }
    //calcul résultats dernière couche
    for (int i = 0; i < this->nbOut; i++) {
        this->inputs[layer][i] = calculTotalPredictReg(layer, nbNeurons[layer - 1], i, this->inputs[layer - 1]);
    }

    return this->inputs[layer];
}


/**
 *
 * Fonctions permettant d'implémenter le Train
 *
 **/
double PMC::calculTotalSigma(int layer, int nbOutput, int input, double *sigma) {
    double total = 0;
    for (int output = 0; output < nbOutput; output++) {
        total += this->weights[layer][output][input] * sigma[output];
    }
    return total;
}

void PMC::calcul_sigma(double **sigma) {
    //calcul
    double total = 0;
    for (int i = 0; i < this->nbNeurons[layer - 1]; i++) {
        total = calculTotalSigma(this->layer, this->nbOut, i, sigma[layer]); //total avant dernière couche
        sigma[layer - 1][i] = (1.0 - (this->inputs[layer - 1][i] * this->inputs[layer - 1][i])) * total;
    }

    for (int l = this->layer - 1; l > 0; l--) {
        for (int i = 0; i < this->nbNeurons[l - 1]; i++) {
            total = calculTotalSigma(l, this->nbNeurons[l], i, sigma[l]);
            sigma[l - 1][i] = (1.0 - (this->inputs[l - 1][i] * this->inputs[l - 1][i])) * total; //total couches précédentes
        }
    }
}

void PMC::train(int epochs, double LR, double **points, double **Y, int pointsSize) {
    //initialisation sigma[]
    double **sigma = new double *[layer + 1];
    for (int i = 0; i < layer + 1; i++) {
        if (i == layer) {
            sigma[i] = new double[nbOut];
        } else {
            sigma[i] = new double[nbNeurons[i] + 1];
        }
    }

    //Itérations sur le batch
    for (int epoch = 0; epoch < epochs; epoch++) {
        int point = rand() % pointsSize;
        double *result = predict(points[point]);


        //calcul sigma dernière couche
        for (int i = 0; i < this->nbOut; i++) {
            sigma[layer][i] = (1 - (result[i] * result[i])) * (result[i] - Y[point][i]);
        }
        //calcul des autres sigma
        calcul_sigma(sigma);


        //mise à jour des poids
        //couche 0
        for (int output = 0; output < nbNeurons[0]; output++) {
            this->weights[0][output][0] -= LR * sigma[0][output];
            updateWeights(0, nbEntry, output, points[point], sigma[0][output], LR);
        }
        //autres couches
        for (int l = 1; l < this->layer; l++) {
            for (int output = 0; output < nbNeurons[l]; output++) {
                this->weights[l][output][0] -= LR * sigma[l][output];
                updateWeights(l, this->nbNeurons[l - 1], output, this->inputs[l - 1], sigma[l][output], LR);
            }
        }
        //dernière couche
        for (int output = 0; output < this->nbOut; output++) {
            this->weights[layer][output][0] -= LR * sigma[layer][output];
            updateWeights(layer, this->nbNeurons[layer - 1], output, this->inputs[layer - 1], sigma[layer][output], LR);
        }

    }
}

void PMC::trainReg(int epochs, double LR, double **points, double **Y, int pointsSize) {
    //initialisation sigma[]
    double **sigma = new double *[layer + 1];
    for (int i = 0; i < layer + 1; i++) {
        if (i == layer) {
            sigma[i] = new double[nbOut];
        } else {
            sigma[i] = new double[nbNeurons[i] + 1];
        }
    }

    //Itérations sur le batch
    for (int epoch = 0; epoch < epochs; epoch++) {
        int point = rand() % pointsSize;
        double *result = predict(points[point]);


        //calcul sigma dernière couche
        for (int i = 0; i < this->nbOut; i++) {
            sigma[layer][i] = result[i] - Y[point][i];
        }
        //calcul des autres sigma
        calcul_sigma(sigma);


        //mise à jour des poids
        //couche 0
        for (int output = 0; output < nbNeurons[0]; output++) {
            this->weights[0][output][0] -= LR * sigma[0][output];
            updateWeights(0, nbEntry, output, points[point], sigma[0][output], LR);
        }
        //autres couches
        for (int l = 1; l < this->layer; l++) {
            for (int output = 0; output < nbNeurons[l]; output++) {
                this->weights[l][output][0] -= LR * sigma[l][output];
                updateWeights(l, this->nbNeurons[l - 1], output, this->inputs[l - 1], sigma[l][output], LR);
            }
        }
        //dernière couche
        for (int output = 0; output < this->nbOut; output++) {
            this->weights[layer][output][0] -= LR * sigma[layer][output];
            updateWeights(layer, this->nbNeurons[layer - 1], output, this->inputs[layer - 1], sigma[layer][output], LR);
        }

    }
}


void PMC::updateWeights(int layer, int nbInput, int output, double *inputs, double sigma, double LR) {
    for (int input = 1; input < nbInput + 1; input++) {
        this->weights[layer][output][input] -= LR * inputs[input - 1] * sigma;
    }
}


/*PMC::~PMC(){
    for(int i = 0; i<layer; i++) {
        for (int j = 0; j < nbNeurons[0]; j++) {
            delete weights[i][j];
        }
        delete weights[i];
    }
    for (int j = 0; j < nbOut; j++) {
        delete weights[layer][j];
    }
    delete weights[layer];
    delete weights;


    for (int i = 0; i < layer + 1; i++) {
        delete inputs[i];
    }
    delete inputs;
}*/




/**
 *
* ToString
 *
 **/

void PMC::tostring() {
    std::string result = "";
    for (int i = 0; i < this->layer + 1; i++) {
        if (i != layer) {
            for (int j = 0; j < this->nbNeurons[i]; j++) {
                if (i == 0) {
                    result += "Neuron: ";
                    for (int w = 0; w < this->nbEntry + 1; w++) {
                        result += " ";
                        result += std::to_string(this->weights[i][j][w]);
                    }

                } else {
                    result += "Neuron: ";
                    for (int w = 0; w < this->nbNeurons[i - 1] + 1; w++) {
                        result += " ";
                        result += std::to_string(this->weights[i][j][w]);
                        //result += " : ";
                        //result += std::to_string(this->inputs[i][j][w]);
                    }

                }
                result += " [";
                result += std::to_string(this->inputs[i][j]);
                result += "] ";

                result += "      ";
            }
        } else {
            for (int j = 0; j < nbOut; j++) {
                for (int w = 0; w < nbNeurons[i - 1] + 1; w++) {
                    result += " ";
                    result += std::to_string(this->weights[i][j][w]);
                }
                result += " [";
                result += std::to_string(this->inputs[i][j]);
                result += "] ";
            }
        }

        result += "\n\n";
    }
    std::cout << result << std::endl;
}
