import numpy as np
import traceback 
import sys 
import datetime


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import seaborn as sns


from load_data import init_train_images
from load_data import init_test_images
from load_data import get_test_set
from load_data import get_pattern_length
from load_data import flatten

def main():

    #Debug
    np.set_printoptions(threshold=np.inf)


    print("[LOG] Creando set de caracteres - " , datetime.datetime.now())    
    
    train_images = init_train_images()
    test_images = init_test_images()
    
    

    print("=>c032:")
    print(train_images[0])
    print("=>noisy_c032:")
    print(test_images[0][0])
    
    
    print("[LOG] Training -        " , datetime.datetime.now())    
    
    # No of neurons
    n_neurons = get_pattern_length()
    W = train(n_neurons, train_images)
    
    # Test
    print("[LOG] Testing -         " , datetime.datetime.now())    
 
    test_set = get_test_set()
    accuracy, op_imgs = test(W, test_set)

    # Resultados
    print("[LOG] La precision de la red es %f" % (accuracy * 100))
    return

       

    
# El entrenamiento es el producto exterior
# entre un vector de entrada y su transpuesta,
# da como resultado la matriz de pesos W cuya
# diagonal son ceros
# En cada entrenamiento se va sumando la matriz
# W obtenida a la W del entrenamiento anterior
#
# neu: cantidad de elementos de cada vector de entrada
# training data: vector de vectores de datos de entrada
def train(neu, training_data):
    #row, col = training_data.shape
    w = np.zeros([neu, neu])
    for data in training_data:
        #display(data)
        w += np.outer(data, data)
    for diag in range(neu):
        w[diag][diag] = 0

#    print("Matriz de pesos:")
#    for row in w:
#        for val in row:
#            print(val)
#        print

    return w


# Function to test the network
def test(weights, testing_data):
    success = 0.0

    output_data = []

    for data in testing_data:
        true_data = data[0]
        noisy_data = data[1]

        # Utilizando la matriz W, ingreso con el pattern a reconocer
        p = retrieve_pattern(weights, noisy_data)
        predicted_data = convert_to_binary(p)
        
        # Comparo el pattern obtenido con el que deberia ser
        if np.array_equal(true_data, predicted_data):
            success += 1.0
            print("+Eureka!")
        else:
            print("-Failed");
        output_data.append([true_data, noisy_data, predicted_data])

    return (success / len(testing_data)), output_data
        

# Function to retrieve individual noisy patterns
def retrieve_pattern(weights, pattern, steps=10):
    #res = flatten(pattern)
    res = pattern

    print("res length = " + str(len(res)))
    print("weights length = " + str(len(weights)))
    for _ in tqdm(range(steps)):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res
    
def convert_to_binary(a):
    #ca = [1 if x==1 else 0 for x in a]
    ca = [1 if x==True else 0 for x in a]
    return ca


def check(item):
    res = [(type(item), len(item))]
    for i in item:
        res.append((type(i), (len(i) if hasattr(i, '__len__') else None)))
    return res

def display(pattern):
    from pylab import imshow, cm, show
    imshow(pattern.reshape((64,64)),cmap=cm.binary, interpolation='nearest')
    show()

if __name__ == "__main__":
    try: 
        main()
    except: 
        traceback.print_exception(*sys.exc_info())         
        print("[LOG] Error -           " , datetime.datetime.now())         