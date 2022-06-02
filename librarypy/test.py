import ctypes
import os
import new_test

points =[[-1,-1],
         [1,-1],
         [-1,1],
         [1,1]]

Y = [[1,-1],
     [-1,1],
     [-1,1],
     [1,-1]]

nbn = [2]

dll = new_test.create_dll()
model = new_test.instanciateMLPModel(dll, 1, nbn, 2, 2)
print(len(points[0]))

print(new_test.predictPMC(dll,model, points[0], 0))

