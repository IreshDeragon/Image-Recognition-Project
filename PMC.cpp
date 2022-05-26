#include "PMC.h"
#include "PMC.h"
#include <stdlib.h>
#include <string>
#include <iostream>
PMC::PMC(int layer, int *nbNeurons, int nbEntry, int nbOut) {
    this->nbOut = nbOut;

    //Instantiation du tableau de poids
    weights = new double **[layer + 1];
    for (int i = 0; i < layer; i++) {
        weights[i] = new double *[nbNeurons[i]];
        if (i == 0) {
            for (int j = 0; j < nbNeurons[i]; j++) {
                weights[i][j] = new double[nbEntry + 1];
                for (int w = 0; w < nbEntry + 1; w++) {
                    weights[i][j][w] = (double) rand() / RAND_MAX * 2 - 1;
                }
    if(layer!=0){
        weights = new double **[layer + 1];
        //couche 0
        weights[0] = new double *[nbNeurons[0]];
        for (int j = 0; j < nbNeurons[0]; j++) {
            weights[0][j] = new double[nbEntry + 1];
            for (int w = 0; w < nbEntry + 1; w++) {
                weights[0][j][w] = (double) rand() / RAND_MAX * 2 - 1;
            }
        } else {
        }
        //autres couches
        for (int i = 0; i < layer; i++) {
            weights[i] = new double *[nbNeurons[i]];
            for (int j = 0; j < nbNeurons[i]; j++) {
                weights[i][j] = new double[nbNeurons[i - 1] + 1];
                for (int w = 0; w < nbNeurons[i - 1] + 1; w++) {
                    weights[i][j][w] = (double) rand() / RAND_MAX * 2 - 1;
                }
            }
        }
    }
    weights[layer] = new double *[nbOut];
    for (int i = 0; i < nbOut; i++) {
        weights[layer][i] = new double[nbNeurons[layer - 1] + 1];
        for (int j = 0; j < nbNeurons[layer - 1] + 1; j++) {
            weights[layer][i][j] = (double) rand() / RAND_MAX * 2 - 1;
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


    //Instatiation du tableau d'input

    //Instatiation du tableau d'input (spikes)
    inputs = new double *[layer + 1];
    for (int i = 0; i < layer + 1; i++) {
        if (i == 0) {
 PMC::PMC(int layer, int *nbNeurons, int nbEntry, int nbOut) {
    }
}

/**
 *
 * Fonctions permettant d'implémenter le Predict
 *
 **/

double *PMC::predict(double *entry) {
    for (int i = 0; i < layer; i++) {
        if (i == 0) {
            for (int j = 0; j < this->nbNeurons[i]; j++) {
                double total = this->weights[0][j][0];
                for (int t = 1; t < nbEntry + 1; t++) {
                    total += this->weights[i][j][t] * entry[t - 1];
                }
                this->inputs[0][j] = tanh(total);
            }
        } else {
            for (int j = 0; j < this->nbNeurons[i]; j++) {
                double total = this->weights[i][j][0];
                for (int t = 1; t < nbNeurons[i - 1] + 1; t++) {
                    total += this->weights[i][j][t] * this->inputs[i - 1][t - 1];
                }
                this->inputs[i][j] = tanh(total);
            }
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
    double *result = new double[this->nbOut];

    //calcul résultats dernière couche
    for (int i = 0; i < this->nbOut; i++) {
        double total = this->weights[layer][i][0];
        for (int j = 1; j < nbNeurons[layer - 1]; j++) {
            total += this->weights[layer][i][j] * this->inputs[layer - 1][j];
        }
        this->inputs[layer][i] = tanh(total);
        result[i] = tanh(total);
        this->inputs[layer][i] = calculTotalPredict(layer, nbNeurons[layer - 1], i, this->inputs[layer - 1]);
    }

    return result;
    return this->inputs[layer];
}


void PMC::train(int epochs, double LR, double **points, double *Y, int pointsSize, int YSize) {
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

void PMC::train(int epochs, double LR, double **points, double **Y, int pointsSize, int YSize) {
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
        //std::cout<<epoch<<std::endl;
        int point = rand() % pointsSize;
        double *result = predict(points[point]);

        //init sigma
        double **sigma = new double *[layer + 1];
        for (int i = 0; i < layer + 1; i++) {
            if (i == layer) {
                sigma[i] = new double[nbOut];
            } else {
                sigma[i] = new double[nbNeurons[i] + 1];
            }
        }

        //calcul sigma dernière couche
        for (int i = 0; i < this->nbOut; i++) {
            sigma[layer][i] = (1 - (result[i] * result[i])) * (result[i] - Y[point]);
            sigma[layer][i] = (1 - (result[i] * result[i])) * (result[i] - Y[point][i]);
        }
        //calcul des autres sigma
        calcul_sigma(sigma);



        // calcul autres sigmas
        for (int l = this->layer; l > 0; l--) {
            for (int i = 0; i < this->nbNeurons[l - 1]; i++) {
                double total = 0;
                if (l == this->layer) {
                    for (int n = 1; n < this->nbOut + 1; n++) {
                        total += this->weights[l][n - 1][i] * sigma[l][n - 1];
                    }
                } else {
                    for (int n = 1; n < this->nbNeurons[l] + 1; n++) {
                        total += this->weights[l][n - 1][i] * sigma[l][n - 1];
                    }
                }
                sigma[l - 1][i] = (1.0f - (inputs[l - 1][i] * inputs[l - 1][i])) * total;
                //std::cout<<"sigma[l-1][i] : "<< l-1 << "--"<< i<<  "--"<<sigma[l-1][i]<< std::endl;
                //std::cout<<"total : "<<total<< std::endl;
                //std::cout<<"inputs[l-1][i] : "<< l-1 << "--"<< i<<  "--"<<inputs[l-1][i]<< std::endl;
            }
        }

        //mise à jour des poids
        //a garder
        /*for(int l = 0; l < this->layer; l++){
            for(int i  = 0;  i < nbNeurons[l]; i++){
                if(l == 0){
                    this->weights[l][i][0] += LR*sigma[l][i];
                    for(int j = 1; j < this->nbEntry+1; j++){
                        this->weights[l][i][j] += LR*points[point][j-1]*sigma[l][i];
                    }
                }else{
                    this->weights[l][i][0] += LR*sigma[l][i];
                    for(int j = 1; j < this->nbNeurons[l-1]; j++){
                        this->weights[l][i][j] += LR*this->inputs[l][i]*sigma[l][i]; //l+1 i+1
                    }
                }
            }
        }*/

        for (int l = 0; l < this->layer; l++) {
            for (int i = 0; i < nbNeurons[l]; i++) {
                if (l == 0) {
                    this->weights[l][i][0] -= LR * sigma[l][i];
                    for (int j = 1; j < this->nbEntry + 1; j++) {
                        this->weights[l][i][j] -= LR * points[point][j - 1] * sigma[l][i];
                    }
                } else {
                    this->weights[l][i][0] -= LR * sigma[l][i];
                    for (int j = 0; j < this->nbNeurons[l - 1]; j++) {
                        this->weights[l][i][j] -= LR * this->inputs[l][i] * sigma[l][i];
                    }
                }
            }
        //couche 0
        for (int output = 0; output < nbNeurons[0]; output++) {
            this->weights[0][output][0] -= LR * sigma[0][output];
            updateWeights(0, nbEntry, output, points[point], sigma[0][output], LR);
        }
        for (int i = 0; i < nbNeurons[layer - 1]; i++) {
            for (int j = 0; j < this->nbOut; j++) {
                this->weights[layer][j][i] -= LR * this->inputs[layer][i] * sigma[layer][i];
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
 void PMC::tostring() {
    }
    std::cout << result << std::endl;
}



/*
std::ostream& operator<<(std::ostream &os, const PMC &a){
    std::string  result = "";
    for (int i =0; i< a.layer; i++){
        for(int j = 0; j < a.nbNeurons[i]; j++){
            for(int w = 0; w < a.nbEntry ; w++){
                result += std::to_string(a.weights[i][j][w]);
            }
        }
    }
    os <<result;
    return os;
}*/ 