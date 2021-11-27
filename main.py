import tensorflow as tf
from keras.losses import mse
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense , Conv2D , Flatten ,Conv1D , MaxPooling1D , MaxPool1D
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats.stats import pearsonr
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, LSTM, GRU, Bidirectional, Input, concatenate
from tensorflow.keras.utils import to_categorical

def onehot_encoding(string):
    transtab = str.maketrans('ACGT','0123')
    string= str(string)
    data = [int(x) for x in list(string.translate(transtab))]
    almost = np.eye(4)[data]
    return almost



def tf_pearson(x, y):
    mx = tf.math.reduce_mean(input_tensor=x,keepdims=True)          # E[X]
    my = tf.math.reduce_mean(input_tensor=y,keepdims=True)          # E[Y]
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(input_tensor=tf.multiply(xm,ym))    # E[(X-E[X])*(Y-E[Y])] = COV[X,Y]
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)         # sigma(X)*sigma(Y)
    return  r_num / r_den                                           # COV[X,Y] \ sigma(X)*sigma(Y)


def one_hot_enc(seq):
    seq = seq[:-1]
    seq = seq + "ACGT"
    if 'N' not in seq:
        trans = seq.maketrans('ACGT', '0123')
        numSeq = list(seq.translate(trans))
        return to_categorical(numSeq)[0:-4]
    else:
        trans = seq.maketrans('ACGTN', '01234')
        numSeq = list(seq.translate(trans))
        hotVec = to_categorical(numSeq)[0:-4]
        for i in range(len(hotVec)):
            if hotVec[i][4] == 1:
                hotVec[i] = [0.25,0.25,0.25,0.25,0]
        return np.delete(hotVec,4,1)



def trim_seq(array,how_much):
    halp_p = len(array[0][1])/2
    from_idx = round(halp_p-how_much/2)
    to_idx = round(halp_p+how_much/2)
    trim = array.apply([lambda x :x.str.slice(from_idx,to_idx)])
    return trim

def get_data(path, min_read=2000,add_RNAplfold =True):
    # train
    with open(path+  "/seq/train-seq") as source:
        X_train = np.array(list(map(one_hot_enc, source)))
    y_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['rsr']).to_numpy()
    w_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['c_read']).to_numpy() + \
              pd.read_csv(path + '/csv_data/train_data.csv', usecols=['t_read']).to_numpy()
    chr_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['chromosome']).to_numpy()
    pos_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['position']).to_numpy()
    strand_train = pd.read_csv(path + '/csv_data/train_data.csv', usecols=['strand']).to_numpy()
    # validation
    with open(path+  "/seq/val-seq") as source:
        X_val =  np.array(list(map(one_hot_enc, source)))
        y_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['rsr']).to_numpy()
    w_val = pd.read_csv(path + '/csv_data/val_data.csv', usecols=['c_read']).to_numpy() +\
            pd.read_csv(path + '/csv_data/val_data.csv', usecols=['t_read']).to_numpy()
    # test
    with open(path+  "/seq/test-seq") as source:
        X_test = np.array(list(map(one_hot_enc, source)))
        y_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['rsr']).to_numpy()
    w_test = pd.read_csv(path + '/csv_data/test_data.csv', usecols=['c_read']).to_numpy() + \
             pd.read_csv(path + '/csv_data/test_data.csv', usecols=['t_read']).to_numpy()

    # set val min read
    ids = np.argwhere(w_val > min_read)[:, 0]
    X_val = X_val[ids]
    y_val = y_val[ids]
    w_val = w_val[ids]

    if (add_RNAplfold):
        X_new = []
        for i in range(len(X_train)):
            plfold_name = ""
            a = str(chr_train[i])[2:-2]
            b = str(pos_train[i] - 140)[1:-1]
            c = str(pos_train[i] + 110)[1:-1]
            d = str(strand_train[i])[2:-2]
            e = ')_lunp\_clean'
            plfold_name = a + '_' + b + '-' + c + '(' + d + e
            with open("./plfold_files/" + plfold_name) as source:
                pl_train = np.array(list(source))
            temp = X_train[i]
            temp = np.column_stack((temp, pl_train))
            X_new.append(temp)
        X_new = np.array(X_new)

    # scale_labels
    y_train = np.log(y_train)
    y_test = np.log(y_test)
    y_val = np.log(y_val)

    return [X_train, y_train, w_train], [X_test, y_test, w_test], [X_val, y_val, w_val]


# if __name__ == '__main__':
    #  Handle Data


def trim_mat(data,INPUT_SIZE):

    [X_Data, Y_Data, W_Data] = data
    total_data_size = X_Data.shape[1]
    start = total_data_size // 2 - INPUT_SIZE // 2
    end = start + INPUT_SIZE
    X_Data = X_Data[:, start:end, :]
    return data

# raw_labels = pd.read_csv("train_labels.txt",sep="\t", header=None)
# raw_data   = pd.read_csv("train-seq.txt",sep="\t", header=None)
# raw_val_data   = pd.read_csv("val-seq-filtered2000.txt",sep="\t", header=None)
# raw_val_labels = pd.read_csv("val-labels-filtered2000.txt",sep="\t",header=None)





# Results Data frame :
Results_pd = pd.DataFrame(columns = ['INPUT_SIZE','FILTER','KERNEL_SIZE','POOLING','POOL_SIZE','DENCE_1','DENCE_2','ACTIVATION_1','ACTIVATION_2','DROPOUT_1','DROPOUT_2',"TF_SEED",'BATCH_SIZE','EPOCH','CONV_PADDING','loss mse','pearson correlation'])
Cloud_run = True
ITER      = 1000


if Cloud_run :
    runidx = "maor_parmas_seed_scan_2"
    iterations = ITER
    path = "/home/u110379/RG4_Proj/rg4_data"
    VERBOSE = 0
else:
    runidx = "debug"
    iterations = 1
    path = "./rg4_data"
    VERBOSE = 1

train, test, validation = get_data(path,add_RNAplfold=False)

for i in range(iterations) :
### Random initilization

    # Initial Values -- Person 0.7034
    # INPUT_SIZE       = 60
    # FILTER           = 88
    # KERNEL_SIZE      = 8
    # POOLING          = 1
    # POOL_SIZE        = 4
    # DENCE_1          = 52
    # DENCE_2          = 40
    # ACTIVATION_1     = 'relu'
    # ACTIVATION_2     = 'relu'
    # DROPOUT_1        = 0.3
    # DROPOUT_2        = 0.1
    # TF_SEED          = 9
    # EPOCH            = 10
    # BATCH_SIZE       = 64
    # CONV_PADDING     = "valid"

    # Maors Params : person = 0.657
    INPUT_SIZE       = 100
    FILTER     = 32
    KERNEL_SIZE= 42
    POOLING = 1
    POOL_SIZE = 45
    DENCE_1 = 64
    DENCE_2 = 16
    ACTIVATION_1 = 'relu'
    ACTIVATION_2 = 'relu'
    DROPOUT_1        = 0.3
    DROPOUT_2        = 0.1
    TF_SEED          = i
    EPOCH            = 10
    BATCH_SIZE       = 64
    CONV_PADDING     = "valid"

# Maors params 2

    # INPUT_SIZE = 140
    # FILTER = 128
    # KERNEL_SIZE = 5
    # POOLING = 7
    # POOL_SIZE = 45
    # DENCE_1 = 128
    # DENCE_2 = 32
    # ACTIVATION_1 = 'relu'
    # ACTIVATION_2 = 'relu'


    # INPUT_SIZE           = np.random.randint(4,11)*10
    # FILTER         = np.random.randint(1,100)
    # KERNEL_SIZE    = np.random.randint(3,INPUT_SIZE/4)
    # POOLING        = np.random.randint(0,2)
    # DENCE_1          = np.random.randint(4,16)*4
    # DENCE_2          = np.random.randint(4,16)*4
    # activation = ['relu','sigmoid','softmax','relu']
    # ACTIVATION_1 = activation[np.random.randint(0,4)] # 50% relu ,25 % sigmoid , 25% softmax
    # ACTIVATION_2 = activation[np.random.randint(0,4)] # 50% relu ,25 % sigmoid , 25% softmax
    # POOL_SIZE   = np.random.randint(2,9)

### End of random initilization
    Results_pd.at[i,'INPUT_SIZE'] = INPUT_SIZE
    Results_pd.at[i,'FILTER'] = FILTER
    Results_pd.at[i,'KERNEL_SIZE'] = KERNEL_SIZE
    Results_pd.at[i,'POOLING'] = POOLING
    Results_pd.at[i,'POOL_SIZE'] = POOL_SIZE
    Results_pd.at[i,'DENCE_1'] = DENCE_1
    Results_pd.at[i,'DENCE_2'] = DENCE_2
    Results_pd.at[i,'ACTIVATION_1'] = ACTIVATION_1
    Results_pd.at[i,'ACTIVATION_2'] = ACTIVATION_2
    Results_pd.at[i,'DROPOUT_1'] = DROPOUT_1
    Results_pd.at[i,'DROPOUT_2'] = DROPOUT_2
    Results_pd.at[i,'TF_SEED'] = TF_SEED
    Results_pd.at[i,'EPOCH'] = EPOCH
    Results_pd.at[i,'BATCH_SIZE'] = BATCH_SIZE
    Results_pd.at[i,'CONV_PADDING'] = CONV_PADDING

## Trim Data acording to randomization :
    train      = trim_mat(train,INPUT_SIZE)
    test       = trim_mat(test,INPUT_SIZE)
    validation = trim_mat(validation,INPUT_SIZE)

    [X_train, Y_train, W_train] = train
    [X_validation, Y_validation, W_validation] = validation
    [X_test, Y_test, W_test] = test
    data_shape = X_train.shape[1:]

    # new comment i am adding to the code 
    
    # trim_data_val = trim_seq(raw_val_data,INPUT_SIZE)
    # trim_data_seq_val = trim_data_val[0].stack()
    # data_val = np.array(list(map(onehot_encoding, trim_data_seq_val)))
    #
    # labels_val = np.array(raw_val_labels[0])
    # labels_val = np.log(labels_val)
    #
    # trim_data = trim_seq(raw_data,INPUT_SIZE)
    # trim_data_seq = trim_data[0].stack()
    # data = np.array(list(map(onehot_encoding, trim_data_seq)))
    # labels = np.array(raw_labels[0])
    # labels = np.log(labels)


## Build model

    tf.random.set_seed(TF_SEED)
    # model = Sequential()
    # model.add(Conv1D(filters=FILTER, kernel_size=KERNEL_SIZE, strides=1, activation='relu', input_shape=(INPUT_SIZE,4),use_bias=True))
    # if POOLING == 1:
    #     model.add(MaxPooling1D(pool_size=POOL_SIZE))
    # model.add(Dropout(0.4))
    # model.add(Flatten())
    # model.add(Dense(DENCE_1, activation = ACTIVATION_1))
    # model.add(Dropout(0.2))
    # model.add(Dense(DENCE_2, activation=ACTIVATION_2))
    # model.add(Dense(1, activation='linear'))
    # model.summary()
    # model.compile(optimizer =Adam(learning_rate=0.001),loss ='mse')




    model = Sequential()
    model.add(Conv1D(filters=FILTER, kernel_size=KERNEL_SIZE, input_shape=data_shape, name="conv",padding=CONV_PADDING))
    model.add(MaxPool1D(pool_size=POOL_SIZE, name="pooling"))
    model.add(Dropout(DROPOUT_1))
    model.add(Flatten())
    model.add(Dense(DENCE_1, activation='relu', name="dense"))
    model.add(Dropout(DROPOUT_2))
    model.add(Dense(DENCE_2, activation='relu', name="dense2"))
    model.add(Dense(1, activation='linear', name="1dense"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    if not Cloud_run :
        model.summary()



    # for i in range(10) :
    model.fit(x=X_train,y=Y_train,epochs=EPOCH,verbose=VERBOSE,batch_size=BATCH_SIZE)
    pred_val = model.predict(X_validation)
    pred_val = pred_val.reshape(len(pred_val))
    Y_validation = Y_validation.reshape(len(Y_validation))

    pred_test = model.predict(X_test)
    pred_test = pred_test.reshape(len(pred_test))
    Y_test = Y_test.reshape(len(Y_test))
    x = pearsonr(Y_validation, pred_val)
    Results_pd.at[i,'pearson correlation'] = pearsonr(Y_validation, pred_val)



    Results_pd.at[i,'loss mse']            = np.mean(np.square(Y_validation - pred_val))
    if not Cloud_run :
        print("pearson correlation :" + str(Results_pd.at[i,'pearson correlation']))
        print("mse :" + str(Results_pd.at[i,'loss mse']))


if Cloud_run :
    filename_csv = "out_results_csv" + "_" + str(runidx)
    filename_excel   = "out_results_excel" + "_" + str(runidx) + ".xlsx"
    Results_pd.to_excel(filename_excel)
    Results_pd.to_csv(filename_csv)
    print("Done Exporting CSV and Excel Result Outputs")
