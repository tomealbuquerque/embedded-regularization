from math import ceil
from tensorflow.keras.applications import resnet50, vgg16, densenet, inception_resnet_v2, inception_v3
from tensorflow.keras import utils, callbacks, models, layers, regularizers
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import generator_heatmaps
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import _pickle as cPickle



# GPU Options
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


LR = 0.0001  # default=0.001
TRAIN_PART = False 
EPOCHS = 17
SEED = 1
BATCH_SIZE=32


#define loss

loss1 = tf.keras.losses.BinaryCrossentropy()


lambda_l=0.01


def gaussian_square(size, fwhm=3):
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

h=1-gaussian_square(28, 28)
w = h.reshape((1, 28, 28, 1))

import matplotlib.pyplot as plt

plt.imshow(h)

peso= 0.5

def loss2(y,act):
    return ((w*act)*lambda)

#Stop when val_loss is not decreasing
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')


######################### Data Augmentation #############################

IMAGE_HEIGHT=224
IMAGE_WIDTH=224
IMAGE_CHANNEL=3
SEED = 1


#Generator ARGS
d=pd.read_pickle('nci_dados.pkl')
pdir = 'preprocess_imgs'

# CV Fold Indices
# Train Indices
with open('train_indices_list.pickle', 'rb') as tr:
    train_indices_list = cPickle.load(tr)

# Validation Indices
with open('val_indices_list.pickle', 'rb') as vl:
    val_indices_list = cPickle.load(vl)

# Test Indices
with open('test_indices_list.pickle', 'rb') as ts:
    test_indices_list = cPickle.load(ts)


dados=pd.DataFrame(d).to_numpy()

for fold in range(4,5):
    print("Current fold: {}".format(fold+1))
    # name='teste'

    d_train=d.iloc[train_indices_list[fold],:]
    d_val=d.iloc[val_indices_list[fold],:]
    d_test=d.iloc[test_indices_list[fold],:]
    
    nci_gen = generator_heatmaps.Generator(d_train, pdir,use_data_aug=True)
    nci_val_gen  = generator_heatmaps.Generator(d_val, pdir)
    nci_test_gen  = generator_heatmaps.Generator(d_test, pdir, batchsize=1)

    # Class Weights
    y_train = d_train['label'].values
    weights=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
  
    
    ############################# Model ##################################### 

        
    vgg = tf.keras.applications.VGG16(False)
    clinical_input = tf.keras.layers.Input(shape=[4])
    
    x = vgg.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Concatenate()([x, clinical_input])
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(2, name='labels', activation='sigmoid')(x)
    act = vgg.get_layer('block3_pool').output
    model = tf.keras.models.Model([vgg.input, clinical_input], [x, act])
    #model.load_weights('gaussian_fold_4_block_05peso_17.h5')
    model.compile(Adam(lr=LR),[loss1,loss2], ['accuracy'] )
    model.summary()
    # tf.keras.utils.plot_model(model, 'model.png', True, False)
    opt = tf.keras.optimizers.Adam(lr=LR)

    
    #################################### TRAIN AND TEST ########################################
    name = 'fold%d-lr%s%s' % (fold, str(LR), '-trainpart' if TRAIN_PART else '-trainall')
    
    
    #Save the model after every epoch.
    checkpointer = ModelCheckpoint(filepath='gaussian_fold_'+str(fold)+'_block_05peso_'+str(EPOCHS)+'.h5', verbose=2, save_best_only=True)


    # # int(nci_gen.size_train/nci_gen.batchsize)
     history = model.fit_generator(
                         nci_gen.generate(), steps_per_epoch=int(nci_gen.size_train/nci_gen.batchsize),
                         epochs=EPOCHS,verbose=1,
                         # class_weight=[weights,weights],
                         validation_data=nci_val_gen.generate(),
                         validation_steps=int(nci_val_gen.size_train/nci_val_gen.batchsize),
                         callbacks=[checkpointer, earlyStopping]
                         )
    

    # model_test = load_model('gaussian_fold_0_block_05peso_53.h5',compile=None)
    Yn_hat,_ = model.predict_generator(nci_test_gen.generate(),steps=d_test['label'].size,verbose=1)
    score_pred = Yn_hat[:,1]
    y_true = d_test['label'].values
    auc=str(roc_auc_score(y_true, score_pred))
    confm=str(confusion_matrix(y_true, score_pred.round()))
    print(confm)
    


    f = open('results_vgg_gaussian.txt', 'a+')
    f.write('\n\nNCI_heatmaps_224x224_fold_%d:' % fold +
        ' Confusion Matrix:'+ confm+
        ' \nAUC:'+ auc+
        '\n Accuracy:'+str(accuracy_score(y_true, score_pred.round()))+
        '\n Balanced acc:'+ str(balanced_accuracy_score(y_true, score_pred.round())))
    
    f.close()
