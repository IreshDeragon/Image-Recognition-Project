#include <cstdint>
#include "PMC.h"

#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllexport)
#endif

extern "C" {

    DLLEXPORT PMC *createMLPModel(int32_t layerPy, int32_t *nbNeuronsPy, int32_t nbEntryPy, int32_t nbOutPy) {
        PMC* model = new PMC(layerPy, nbNeuronsPy, nbEntryPy, nbOutPy);
        return model;
    }

    DLLEXPORT  void trainPMC(PMC *model, double **points, int32_t pointsSize, double **Y, bool is_classification, double LR, int32_t epochs) {
        model->train(epochs, LR, points, Y,  pointsSize);
    }

    DLLEXPORT     double *predictPMC(PMC *model, double *points, bool is_classification) {
        return model->predict(points);
    }

    DLLEXPORT void destroyPMCModel(PMC *model) {
        //delete model;
    }
}
