import numpy as np
import traceback 
import sys 
import datetime

from training_data import init_train_images
from training_data import init_test_images


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
    
    
    # Size of image(width)
    n_side = 64
    
    print("[LOG] Training -        " , datetime.datetime.now())    
    
    # No of neurons
    n_neurons = n_side * n_side
    W = train(n_neurons, train_images)
    
    # Test
    print("[LOG] Testing -         " , datetime.datetime.now())    
 
    accuracy, op_imgs = test(W, test_images)

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
    w = np.zeros([neu, neu])
    for data in training_data:
        w += np.outer(data, data)
    for diag in range(neu):
        w[diag][diag] = 0
    print("Matriz de pesos:")
    for row in w:
        for val in row:
            print(val)
        print

    return w


# Function to test the network
def test(weights, testing_data):
    success = 0.0

    output_data = []

    for data in testing_data:
        true_data = data[0]
        #noisy_data = np.array(data[1])
        noisy_data = convert_to_binary(data[1])

        print("**************");
        print("Con ruido:")
        print(noisy_data)
        print("Esperado:")
        print(true_data)
        
        # Utilizando la matriz W, ingreso con el pattern a reconocer
        #predicted_data = retrieve_pattern(weights, noisy_data)
        predicted_data = convert_to_binary(retrieve_pattern(weights, noisy_data))
        
        # Comparo el pattern obtenido con el que deberia ser
        print("Obtenido:")
        print(predicted_data)
        print("**************");
        if np.array_equal(true_data, predicted_data):
            success += 1.0
            print("=>Eureka!")
        output_data.append([true_data, noisy_data, predicted_data])

    return (success / len(testing_data)), output_data
        

# Function to retrieve individual noisy patterns
def retrieve_pattern(weights, pattern, steps=10):
    #res = np.array(pattern.copy())
    res = pattern.copy()
    #res = pattern

    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res
    
def convert_to_binary(a):
    ca = [1 if x==1 else 0 for x in a]
    #ca = [1 if x==1 else x for x in a]
    #ca = [0 if x==-1 else x for x in a]
    return ca

def convert_to_bipolar(a):
    ca = [1 if x==1 else -1 for x in a]
    return ca


if __name__ == "__main__":
    try: 
        main()
    except: 
        traceback.print_exception(*sys.exc_info())         
        print("[LOG] Error -           " , datetime.datetime.now())         