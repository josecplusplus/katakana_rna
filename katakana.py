import numpy as np
import traceback 
import sys 
import datetime


def main():

    print("[LOG] Creando set de caracteres - " , datetime.datetime.now())    
    
    # Creo bitmaps de 64 bits
    
    # Bitmaps obtenidos de https://www.thingiverse.com/thing:10195/files

    # Caracter "ka"
    ka = [  0,0,0,0,0,0,0,0,
			0,0,1,0,0,0,0,0,
			0,1,1,1,1,1,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,0,0,0,0,0,0]

    # Caracter "ki"
    ki = [  0,0,0,0,0,0,0,0,
			0,0,1,0,0,0,0,0,
			0,1,1,1,1,1,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,0,0,0,0,0,0]

    # Caracter "ku"
    ku = [  0,0,0,0,0,0,0,0,
			0,0,1,0,0,0,0,0,
			0,1,1,1,1,1,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,0,0,0,0,0,0]
    
    # Caracter "ke"
    ke = [  0,0,0,0,0,0,0,0,
			0,0,1,0,0,0,0,0,
			0,1,1,1,1,1,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,1,0,0,0,1,0,
			0,0,0,0,0,0,0,0]
    
    
    train_images = [ka, ki, ku, ke]

    # Size of image(width)
    n_side = 8
    
    print("[LOG] Training -        " , datetime.datetime.now())    
    
    # No of neurons
    n_neurons = n_side * n_side
    W = train(n_neurons, train_images)
    
    # Test
    print("[LOG] Testing -         " , datetime.datetime.now())    

    # Creo un set de datos, donde las posiciones pares 
    # son los caracteres originales, y las impares
    # son los caracteres con ruido
    
    noisy_ka = ka.copy()
    noisy_ki = ki.copy()
    noisy_ku = ku.copy()
    noisy_ke = ke.copy()
    
    
    test_images = [[ka, noisy_ka], [ki, noisy_ki], [ku, noisy_ku], [ke, noisy_ke]]
    
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
    return w


# Function to test the network
def test(weights, testing_data):
    success = 0.0

    output_data = []

    for data in testing_data:
        true_data = data[0]
        noisy_data = data[1]
        
        # Utilizando la matriz W, ingreso con el pattern a reconocer
        predicted_data = retrieve_pattern(weights, noisy_data)
        
        # Comparo el pattern obtenido con el que deberia ser
        if np.array_equal(true_data, predicted_data):
            success += 1.0
        output_data.append([true_data, noisy_data, predicted_data])

    return (success / len(testing_data)), output_data
        

# Function to retrieve individual noisy patterns
def retrieve_pattern(weights, data, steps=10):
    res = np.array(data)

    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res

if __name__ == "__main__":
    try: 
        main()
    except: 
        traceback.print_exception(*sys.exc_info())         
        print("[LOG] Error -           " , datetime.datetime.now())         