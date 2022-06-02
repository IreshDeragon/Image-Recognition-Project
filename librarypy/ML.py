import ctypes
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt


class MLPModel():

    def __init__(self, layer, nbNeurons,nbEntry, nbOut):
        self.dll = self.load_dll("libMLlibrary.dll") #Path de la librairie c++
        self.model = self.instanciateMLPModel(layer, nbNeurons, nbEntry, nbOut)



    def instanciateMLPModel(self, layer, nbNeurons, nbEntry, nbOut):
        flattened_nbNeurons = list(itertools.chain(nbNeurons))
        flattened_nbNeurons_type = ctypes.c_float * len(flattened_nbNeurons)
        Native_flattened_nbNeurons = flattened_nbNeurons_type(*flattened_nbNeurons)

        MLPmodelType = ctypes.POINTER(ctypes.c_char)
        print(type(self))
        self.dll.createMLPModel.argtypes = [ ctypes.c_int32, flattened_nbNeurons_type,
                                            ctypes.c_int32, ctypes.c_int32]
        self.dll.createMLPModel.restype = ctypes.c_void_p #ctypes.Structure
        #native_nbNeurons_array = nbNeurons_type * len(nbNeurons)
        return self.dll.createMLPModel(layer, Native_flattened_nbNeurons, nbEntry, nbOut)




    def trainPMC(self, points, Y, LR, epochs):

        flattened_Y = list(itertools.chain(*Y))
        flattened_points = list(itertools.chain(*points))

        flattened_points_type = ctypes.c_float * len(flattened_points)
        Native_flattened_points = flattened_points_type(*flattened_points)

        flattened_Y_type = ctypes.c_float * len(flattened_Y)
        Native_flattened_Y = flattened_points_type(*flattened_Y)

        self.dll.train_mlp_model.argtypes = [ctypes.c_void_p, flattened_points_type, flattened_Y_type, ctypes.c_float, ctypes.c_int32]
        self.dll.train_mlp_model.restype = None

        self.dll.train_mlp_model(self.model, Native_flattened_points,
                               Native_flattened_Y, LR, epochs)



    def predictPMC(self, model, pointsk):
        predict_points_type = ctypes.c_float * len(pointsk)
        Native_predict_points = ctypes.c_float(*pointsk)

        self.dll.predict_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, predict_points_type]
        self.dll.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_float)

        result = self.dll.predict_mlp_model(model, Native_predict_points)

        return np.ctypeslib.as_array(result)

    def destroyPMCModel(self):
        self.dll.destroyPMCModel.restype = None

    def load_dll(self, param):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dll_file = os.path.join(dir_path, param)

        lib = ctypes.CDLL(dll_file)
        return lib