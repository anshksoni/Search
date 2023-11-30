
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow.keras as keras
import tensorrt
import gc
import numpy as np

def small_CNN():
    input     = layers.Input(shape=(224, 224,1))
    conv1     = layers.Conv2D(8, (3, 3),strides=(3,3), activation="relu", padding="same")(input)
    batch1    = layers.BatchNormalization(axis=-1)(conv1)
    conv2     = layers.Conv2D(10, (3, 3),strides=(3,3), activation="relu", padding="same")(batch1)
    conv3     = layers.Conv2D(15, (3,3),strides=(3,3), activation="relu", padding="same")(conv2)
    flatten1  = layers.Flatten()(conv3)
    dense1    = layers.Dense(256,activation="relu" )(flatten1)
    out       = layers.Dense(2,activation="softmax" )(dense1)

    Classifier = Model(input, out,name="Classifier")
    toptimizer = keras.optimizers.experimental.AdamW()
    Classifier.compile(loss='BinaryCrossentropy', optimizer=toptimizer,metrics=['accuracy'])
    Classifier.summary()
    return Classifier

def VGG_for_transfer():
    vgg16_model=keras.applications.vgg16.VGG16(

        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )
    VGG16= keras.Sequential()
    for layer in vgg16_model.layers[:20]:
        VGG16.add(layer)
    for layer in VGG16.layers:
        layer.trainable = False
    VGG16.add(layers.BatchNormalization(axis=-1))
    VGG16.add(layers.Dense(4096, activation='ReLU', name='fc1'))
    VGG16.add(layers.Dense(4096, activation='ReLU', name='fc2'))
    VGG16.add(layers.Dense(2, activation='softmax', name='predictions'))
    toptimizer = keras.optimizers.experimental.AdamW()
    VGG16.compile(loss='BinaryCrossentropy', optimizer=toptimizer,metrics=['accuracy'])
    VGG16.summary()
    return VGG16


def train_Classifier(Classifier,trainfunc,testfunc,target_noise):
    for i in range(2):
        train_set, target_present = trainfunc(2500,0)
        test_set, target_present_test = testfunc(4,500,0)
        epochs=20
        H = Classifier.fit(train_set, target_present,validation_data=(test_set, target_present_test),epochs=epochs,batch_size=50)
        del train_set
        del target_present_test
        del test_set
        del target_present
        gc.collect()
    gc.collect()

    for noises in  [target_noise/5,3*target_noise/10,2*target_noise/5,target_noise/2,3*target_noise/5,4*target_noise/5,target_noise,target_noise,target_noise,target_noise,target_noise,target_noise]:
        
        for i in range(6):
            train_set, target_present = trainfunc(2500,noises)
            test_set, target_present_test = testfunc(4,500,noises)
            epochs=5
            H = Classifier.fit(train_set, target_present,validation_data=(test_set, target_present_test),epochs=epochs,batch_size=50)
            del train_set
            del target_present_test
            del test_set
            del target_present
            gc.collect()


            print('times: ',i)
    gc.collect()
    return Classifier

def train_large(VGG16,trainfunc,testfunc,target_noise):

    for i in range(2):
        train_set, target_present = trainfunc(2500,0)
        test_set, target_present_test = testfunc(4,500,0)
        epochs=20
        H = VGG16.fit(np.concatenate((train_set,train_set,train_set),axis=3), target_present,validation_data=(np.concatenate((test_set,test_set,test_set),axis=3), target_present_test),epochs=epochs,batch_size=50)
        del train_set
        del target_present_test
        del test_set
        del target_present
        gc.collect()


    for noises in  [target_noise/5,3*target_noise/10,2*target_noise/5,target_noise/2,3*target_noise/5,4*target_noise/5,target_noise,target_noise,target_noise,target_noise,target_noise,target_noise]:
        print(noises)
        for i in range(6):
            train_set, target_present = trainfunc(2500,noises)
            test_set, target_present_test = testfunc(4,500,noises)
            epochs=5
            H = VGG16.fit(np.concatenate((train_set,train_set,train_set),axis=3), target_present,validation_data=(np.concatenate((test_set,test_set,test_set),axis=3), target_present_test),epochs=epochs,batch_size=50)
            del train_set
            del target_present_test
            del test_set
            del target_present
            gc.collect()


            print('times: ',i)
        gc.collect()
    return VGG16




def testSmall(tmod,testsetsizes,noise,n10000,testfunc):
    perfs=[]
    for k in range(n10000):
        perfs_this_set=[]
        for i in testsetsizes:
            a,b=testfunc(i,5000,noise)
            perfs_this_set.append(tmod.evaluate(a,b,batch_size=1)[1])
            keras.backend.clear_session()
            gc.collect()
        perfs.append(perfs_this_set)
    if len(np.mean( np.array(perfs), axis=0 ))==len(testsetsizes):
        return np.mean( np.array(perfs), axis=0 )
    else:
        return np.array(perfs)




def testLarge(tmod,testsetsizes,noise,n10000, testfunc):
    perfs=[]
    for k in range(n10000):
        perfs_this_set=[]
        for i in testsetsizes:
            a,b=testfunc(i,5000,noise)
            a=np.concatenate((a,a,a),axis=3)
            perfs_this_set.append(tmod.evaluate(a,b,batch_size=1)[1])
            keras.backend.clear_session()
            gc.collect()
        perfs.append(perfs_this_set)
    if len(np.mean( np.array(perfs), axis=0 ))==len(testsetsizes):
        return np.mean( np.array(perfs), axis=0 )
    else:
        return np.array(perfs)