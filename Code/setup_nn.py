# import modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
from master_nn import master_nn

######################################################################
def read_mnist_csv():

    path = '/Users/xl332/Desktop/DataProjects/NeuNet/MNIST_digits/'
    data_train = pd.read_csv(path+'mnist_train.csv', skipinitialspace=True)
    data_test = pd.read_csv(path+'mnist_test.csv', skipinitialspace=True)

    data = data_train.append(data_test,ignore_index=True)
    data.dropna(inplace=True)
    data.reset_index(drop=True)

    y = data.iloc[:,0]
    XT = data.iloc[:,1:]
    YTn = one_hot_encoder(y)
    YT = pd.DataFrame(data=YTn)
    XT = XT/255.0
    X = XT.T; Y = YT.T

    return X, Y

def one_hot_encoder(y):

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded

def read_loan_csv():

    fields_X = ['income','age','loan']
    field_Y = ['default']
    data = pd.read_csv('original.csv')

    OT = pd.read_csv('original.csv', skipinitialspace=True)
    OT = OT.loc[OT['age']>0]
    OT = OT.loc[OT['loan']>0]
    OT = OT.loc[OT['income']>0]

    OT.dropna(inplace=True)
    OT.reset_index(drop=True)

    YT = OT[field_Y]
    XT = OT[fields_X]
    XT = (XT - XT.min())/(XT.max()-XT.min())
    X = XT.T; Y = YT.T

    return X, Y

def feed_nn(iter_num,restart):

#    X, Y = read_loan_csv()
    X, Y = read_mnist_csv()

    # split into training and dev sets (additional test desirable if tunning hyperparams)
    pool_frac=0.9 # (0.8: 80% train, 20% dev)
    rand_int = random.sample(range(np.shape(Y)[1]), np.shape(Y)[1])
    n_cut = int(np.floor(pool_frac*np.shape(Y)[1]))
    train_index = rand_int[0:n_cut]
    dev_index = rand_int[n_cut:]
    
    # Y is an array for binary, Y is a one-hot vector for multiclass 
    input_dim = np.shape(X)[0]
    output_dim = np.shape(Y)[0]
    learning_rate     = 0.001 # 0.001 (0.1 for large size batch possible)
    reg_lambd         = 0.0   # default: 0.0 ; recommanded: 0.7
    keepnode_prob     = 1.0   # default: 1.0 ; recommanded: 0.8 
    hidden_activation = 'relu'     # leaky 'relu'
    output_activation = 'softmax'  # 'sigmoid' for binary, 'softmax' for multiclass
    optimizer         = 'adam'     # default: 'gradient_descent', 'adam'
    ###### define architecture (# neurons per layer should be O(input_dim))
    # layer_dims = [input_dim,output_dim]   # L1
    # layer_dims = [input_dim,2,output_dim] # L2
    # layer_dims = [input_dim,3,2,output_dim] # L3
    layer_dims = [input_dim,20,12,output_dim] # L3

    ##### define mini-batch size
#    batch_size = 256 # factor 8 [32,64,128,256,512,1024,2048] 8*(2**N)
    batch_size = 2048 # factor 8 [32,64,128,256,512,1024,2048,4096,8192] 8*(2**N)
    master_nn(X,Y,train_index,dev_index,hidden_activation,output_activation,learning_rate,layer_dims,batch_size,iter_num,optimizer,reg_lambd,keepnode_prob,restart)
