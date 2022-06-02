import ctypes
import itertools
import os
import numpy as np


def create_dll():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dll_file = os.path.join(dir_path, "libMLlibrary.dll")
    dll = ctypes.CDLL(dll_file)
    return dll


def instanciateMLPModel(dll, layer, nbNeurons, nbEntry, nbOut):
    flattened_nbNeurons = list(itertools.chain(nbNeurons))
    flattened_nbNeurons_type = ctypes.c_int32 * len(flattened_nbNeurons)
    Native_flattened_nbNeurons = flattened_nbNeurons_type(*flattened_nbNeurons)

    MLPmodelType = ctypes.POINTER(ctypes.c_char)
    dll.createMLPModel.argtypes = [ctypes.c_int32, flattened_nbNeurons_type,
                                   ctypes.c_int32, ctypes.c_int32]
    dll.createMLPModel.restype = ctypes.c_void_p  # ctypes.Structure
    # native_nbNeurons_array = nbNeurons_type * len(nbNeurons)
    return dll.createMLPModel(layer, Native_flattened_nbNeurons, nbEntry, nbOut)


def trainPMC(dll, points, Y, LR, epochs, PointsSize):
    flattened_Y = list(itertools.chain(*Y))
    flattened_points = list(itertools.chain(*points))

    flattened_points_type = ctypes.c_float * len(flattened_points)
    Native_flattened_points = flattened_points_type(*flattened_points)

    flattened_Y_type = ctypes.c_float * len(flattened_Y)
    Native_flattened_Y = flattened_points_type(*flattened_Y)

    dll.train_mlp_model.argtypes = [ctypes.c_void_p, flattened_points_type, flattened_Y_type, ctypes.c_float,
                                             ctypes.c_int32]
    dll.train_mlp_model.restype = None

    dll.train_mlp_model(epochs,LR, Native_flattened_points,Native_flattened_Y, PointsSize)


def predictPMC(dll, model,pointsk ,isclassification):
    predict_points_type = ctypes.c_float * len(pointsk)

    dll.predictPMC.argtypes = [ctypes.c_void_p, predict_points_type, ctypes.c_bool]
    dll.predictPMC.restype = ctypes.c_void_p

    result = dll.predictPMC(model , pointsk, isclassification)

    return np.ctypeslib.as_array(result)
