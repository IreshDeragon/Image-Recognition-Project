#include "PMC.h"
#include <stdlib.h>

#include <math.h>


PMC::PMC(int32_t layer, int32_t* nbNeurons, int32_t nbEntry, int32_t nbOut) {
    this->layer = layer;
    this->nbNeurons = nbNeurons;
    this->nbEntry = nbEntry;
    this->nbOut = nbOut;

    //Instantiation du tableau de poids
    if (layer != 0) {
        weights = new float **[layer + 1];
        //couche 0
        weights[0] = new float *[nbNeurons[0]];
        for (int j = 0; j < nbNeurons[0]; j++) {
            weights[0][j] = new float[nbEntry + 1];
            for (int w = 0; w < nbEntry + 1; w++) {
                weights[0][j][w] = (float) rand() / RAND_MAX * 2 - 1;
            }
        }
        //autres couches
        for (int i = 0; i < layer; i++) {
            weights[i] = new float *[nbNeurons[i]];
            for (int j = 0; j < nbNeurons[i]; j++) {
                weights[i][j] = new float[nbNeurons[i - 1] + 1];
                for (int w = 0; w < nbNeurons[i - 1] + 1; w++) {
                    weights[i][j][w] = (float) rand() / RAND_MAX * 2 - 1;
                }
            }
        }
        //dernière couche
        weights[layer] = new float *[nbOut];
        for (int i = 0; i < nbOut; i++) {
            weights[layer][i] = new float[nbNeurons[layer - 1] + 1];
            for (int j = 0; j < nbNeurons[layer - 1] + 1; j++) {
                weights[layer][i][j] = (float) rand() / RAND_MAX * 2 - 1;
            }
        }
    } else { //si layer est égal à 0
        weights = new float **[1];
        //couche 0
        weights[0] = new float *[nbOut];
        for (int j = 0; j < nbOut; j++) {
            weights[0][j] = new float[nbEntry + 1];
            for (int w = 0; w < nbEntry + 1; w++) {
                weights[0][j][w] = (float) rand() / RAND_MAX * 2 - 1;
            }
        }
    }



    //Instatiation du tableau d'input (spikes)
    inputs = new float *[layer + 1];
    for (int i = 0; i < layer + 1; i++) {
        if (i == 0) {
            inputs[i] = new float[nbEntry + 1];
        } else if (i == layer) {
            inputs[i] = new float[nbOut + 1];
        } else {
            inputs[i] = new float[nbNeurons[i] + 1];
        }
    }
}

/**
 *
 * Fonctions permettant d'implémenter le Predict
 *
 **/

float PMC::calculTotalPredict(int layer, int nbInput, int output, float *input) {
    float total = this->weights[layer][output][0];
    for (int t = 0; t < nbInput; t++) {
        total += this->weights[layer][output][t + 1] * input[t];
    }
    return tanh(total);
}

float *PMC::predict(float *entry) {
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


/**
 *
 * Fonctions permettant d'implémenter le Train
 *
 **/
float PMC::calculTotalSigma(int layer, int nbOutput, int input, float *sigma) {
    float total = 0;
    for (int output = 0; output < nbOutput; output++) {
        total += this->weights[layer][output][input] * sigma[output];
    }
    return total;
}

void PMC::calcul_sigma(float **sigma) {
    //calcul
    float total = 0;
    for (int i = 0; i < this->nbNeurons[layer - 1]; i++) {
        total = calculTotalSigma(this->layer, this->nbOut, i, sigma[layer]); //total avant dernière couche
        sigma[layer - 1][i] = (1.0 - (this->inputs[layer - 1][i] * this->inputs[layer - 1][i])) * total;
    }

    for (int l = this->layer - 1; l > 0; l--) {
        for (int i = 0; i < this->nbNeurons[l - 1]; i++) {
            total = calculTotalSigma(l, this->nbNeurons[l], i, sigma[l]);
            sigma[l - 1][i] =
                    (1.0 - (this->inputs[l - 1][i] * this->inputs[l - 1][i])) * total; //total couches précédentes
        }
    }
}

void PMC::train(int epoch, float LR, float **points, float **Y, int32_t pointsSize) {
    //initialisation sigma[]
    float **sigma = new float *[layer + 1];
    for (int i = 0; i < layer + 1; i++) {
        if (i == layer) {
            sigma[i] = new float[nbOut];
        } else {
            sigma[i] = new float[nbNeurons[i] + 1];
        }
    }

    //Itérations sur le batch
    for (int epo = 0; epo < epoch; epo++) {
        int point = rand() % pointsSize;
        float *result = predict(points[point]);


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

void PMC::updateWeights(int32_t layer, int32_t nbInput, int32_t output, float *inputs, float sigma, float LR) {
    for (int input = 1; input < nbInput + 1; input++) {
        this->weights[layer][output][input] -= LR * inputs[input - 1] * sigma;
    }
}


PMC::~PMC() {

}


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
