#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "PMC.h"
using namespace std;

int predict(int Xk[], float W[], int size) {
    float total = W[0];
    for (int i = 1; i < size+1; i++) {
        total += W[i] * Xk[i - 1];
    }
    if (total >= 0) {
        return 1;
    } else {
        return -1;
    }
}

void train(float LR, int iter, int** points, int Y[], float *W, int size2 ,int size) {
    float *weights = W;
    for (int o = 0; o < iter; o++) {
        int k = rand() % size;
        int *Xk = points[k];
        int Yk = Y[k];
        for (int i = 0; i < size; i++) {
            if(i == 0){
                weights[i] = weights[i] + LR * (Yk - predict(Xk, weights, size2)) * 1;
            }else {
                weights[i] = weights[i] + LR * (Yk - predict(Xk, weights, size2)) * (Xk[i - 1]);
            }
        }
    }
}

int main() {
    srand(time(NULL));

    // linear classification
    /*int** points = new int *[3];
    for(int i = 0 ; i<4 ; i++){
        points[i] = new int[2];
    }
    points[0][0] = 0;
    points[0][1] = 0;
    points[1][0] = 1;
    points[1][1] = 0;
    points[2][0] = 0;
    points[2][1] = 1;
    points[3][0] = 1;
    points[3][1] = 1;
    //{{0, 0}, {0, 1}, {1, 0}};
    int Y[4] = {-1, 1, 1, 1};

    float W[3];

    int size = sizeof(W) / sizeof(W[0]);
    int size2 = 2;

    for (int i = 0; i < size; i++) {
        W[i] = (float)rand() / RAND_MAX * 2 - 1;
        printf("%f ", W[i]);
    }

    float *narray = W;

    printf(" %d ", predict(points[0] , W, size2));
    printf(" %d ", predict(points[1], W, size2));
    printf(" %d ", predict(points[2], W, size2));
    printf(" %d ", predict(points[3], W, size2));



    train(0.01, 10000, points, Y, narray, 2, size);

    for (int i = 0; i < size; i++) {
        printf("%f ", W[i]);
    }
    printf(" %d ", predict(points[0] , W, size2));
    printf(" %d ", predict(points[1], W, size2));
    printf(" %d ", predict(points[2], W, size2));
    printf(" %d ", predict(points[3], W, size2));*/

    // linear classification


    //PMC

    double** points = new double *[3];
    for(int i = 0 ; i<4 ; i++){
        points[i] = new double[2];
    }
    points[0][0] = 0;
    points[0][1] = 0;
    points[1][0] = 1;
    points[1][1] = 0;
    points[2][0] = 0;
    points[2][1] = 1;
    points[3][0] = 1;
    points[3][1] = 1;

    double Y[4] = {-1, 1, 1, -1};

    int* neurons = new int[1];
    neurons[0] = 2;

    PMC p = PMC(1, neurons, 2, 1);
    p.tostring();
    double* result = p.predict(points[1]);
    for(int i = 0; i < p.nbOut; i++){
        cout << fixed << result[i] << endl;
    }
    p.train(50, 0.1, points, Y, 4, 4);
    p.tostring();

    double* result2 = p.predict(points[1]);
    for(int i = 0; i < p.nbOut; i++){
        cout << fixed << result2[i] << endl;
    }


    /*int cpt = 0;
    for(int i = 0 ; i < p.nbNeurons[0]; i++){
        for(int j = 1; j <= p.nbEntry; j++){
            int cpt2 = 0;
            for(int x = 0; x < 2; x++) {
                p.inputs[0][i][j] = points[cpt][cpt2];
                cpt2++;
            }
        }
        cpt++;
    }
    */
    //PMC
    return 0;
}


//std::cout<<"sigma["<<l<<"]["<<i<<"] = "<<sigma[l][i]<<std::endl;