from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.datasets import mnist

import struct
from PIL import Image
import numpy as np

import skimage.transform
from sklearn.model_selection import train_test_split

import numpy as np

def usage():
    print('''
    Usage: 
        python3 katakana.py

    '''
    )


def main():
    # Cargar datasets ETL1C-07 a ETL1C-13
    read_kana()

    # Transformar
    kana = np.load("kana.npz")['arr_0'].reshape([-1, 63, 64]).astype(np.float32)
    kana = kana/np.max(kana) # make the numbers range from 0 to 1

    # 51 is the number of different katakana (3 are duplicates so in the end there are 48 classes), 1411 writers.
    train_images = np.zeros([51 * 1411, 48, 48], dtype=np.float32)

    for i in range(51 * 1411): # change the image size to 48*48
        train_images[i] = skimage.transform.resize(kana[i], (48, 48))

    arr = np.arange(51) # create labels
    train_labels = np.repeat(arr, 1411)

    # In the actual code, I combined the duplicate classes here and had 48 classes in the end

    # split the images/labels to train and test
    train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2)

    #np.savez_compressed("katakana_train_images.npz", train_images)
    #np.savez_compressed("katakana_train_labels.npz", train_labels)
    #np.savez_compressed("katakana_test_images.npz", test_images)
    #np.savez_compressed("katakana_test_labels.npz", test_labels)    
        
    
    # TODO hacer el training!
    
    # Size of image(width)
    n_side = 48 #FIXME
    
    # No of neurons
    n_neurons = n_side * n_side
    w = train(n_neurons, train_images)
    

    # TODO ejecutar tests


    # TODO evaluar resultados
    
    
    
    
    # load data
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()    
    
    
    # flatten 28*28 images to a 784 vector for each image
#    num_pixels = X_train.shape[1] * X_train.shape[2]
#    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
#    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')    
    
    # normalize inputs from 0-255 to 0-1
#    X_train = X_train / 255
#    X_test = X_test / 255    

    # one hot encode outputs
#    y_train = np_utils.to_categorical(y_train)
#    y_test = np_utils.to_categorical(y_test)
#    num_classes = y_test.shape[1]




    # define baseline model
#    def baseline_model():
        # create model
#        model = Sequential()
#        model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
#        model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
        # Compile model
#        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#        return model
        
    # build the model
#    model = baseline_model()
    # Fit the model
#    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
    # Final evaluation of the model
#    scores = model.evaluate(X_test, y_test, verbose=0)
#    print("Baseline Error: %.2f%%" % (100-scores[1]*100))        
        
        
def read_record_ETL1G(f):
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iL = iF.convert('P')
    return r + (iL,)

def read_kana():
    katakana = np.zeros([51, 1411, 63, 64], dtype=np.uint8) # 51 characters, 1411 writers, img size = 63*64
    for i in range(7,14):
        filename = 'ETL1/ETL1C_{:02d}'.format(i)
        with open(filename, 'rb') as f: # file 13 only has 3 characters, others have 8 characters
            if i!=13: limit = 8
            else: limit=3
            for dataset in range(limit):
                for j in range(1411):
                    try :
                        r = read_record_ETL1G(f)
                        katakana[(i - 7) * 8 + dataset, j] = np.array(r[-1])
                    except struct.error: # two imgs are blank according to the ETL website, so this prevents any errors
                        pass
    np.savez_compressed("kana.npz", katakana)
    
    
def train(neu, training_data):
    w = np.zeros([neu, neu])
    for data in training_data:
        w += np.outer(data, data)
    for diag in range(neu):
        w[diag][diag] = 0
    return w

        

if __name__ == "__main__":
    try: 
        main()
    except ValueError:
        print ("ERROR!")
