int main() {

    //PMC

    double** points = new double *[3];
    for(int i = 0 ; i<4 ; i++){
    double** points = new double *[6];
    for(int i = 0 ; i<6 ; i++){
        points[i] = new double[2];
    }
    points[0][0] = 0;
    points[0][1] = 0;
    points[0][0] = -1;
    points[0][1] = -1;
    points[1][0] = 1;
    points[1][1] = 0;
    points[2][0] = 0;
    points[1][1] = -1;
    points[2][0] = -1;
    points[2][1] = 1;
    points[3][0] = 1;
    points[3][1] = 1;

    double Y[4] = {-1, 1, 1, -1};

    int* neurons = new int[1];
    //double Y[4][2] = {1,-1, 1, -1, 1, -1, -1, 1};
    double** Y = new double *[6];
    for(int i = 0 ; i<6 ; i++){
        Y[i] = new double[2];
    }
    Y[0][0] = 1;
    Y[0][1] = -1;
    Y[1][0] = -1;
    Y[1][1] = 1;
    Y[2][0] = -1;
    Y[2][1] = 1;
    Y[3][0] = 1;
    Y[3][1] = -1;

    //double Y[4][2] = {-1, 1, 1, -1, 1, -1, -1, 1};

    int* neurons = new int[2];
    neurons[0] = 2;

    PMC p = PMC(1, neurons, 2, 1);
    //neurons[0] = 1;
    PMC p = PMC(1, neurons, 2, 2);
    p.tostring();
    double* result = p.predict(points[1]);
    double* result = p.predict(points[3]);
    for(int i = 0; i < p.nbOut; i++){
        cout << fixed << result[i] << endl;
        cout << fixed << "predict :"<<result[i] << endl;
    }
    p.train(50, 0.1, points, Y, 4, 4);
    p.train(1000, 0.1, points, Y, 6, 6);
    p.tostring();

    double* result2 = p.predict(points[1]);
    double* result2 = p.predict(points[3]);
    for(int i = 0; i < p.nbOut; i++){
        cout << fixed << result2[i] << endl;
        cout << fixed <<"predict : " << result2[i] << endl;
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