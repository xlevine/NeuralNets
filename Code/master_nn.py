import numpy as np
from random import sample
######################################################################
def master_nn(X,Y,train_index,dev_index,hidden_activation,output_activation,learning_rate,layer_dims,batch_size,iter_num,optimizer,reg_lambd=0.0,keepnode_prob=1.0,restart='False'):

    dic_hyperparam = {'hidden_activation':hidden_activation,'output_activation':output_activation,'learning_rate':learning_rate,'layer_dims':layer_dims,'train_index':train_index,'dev_index':dev_index,'batch_size':batch_size, 'reg_lambd':reg_lambd,'keepnode_prob':keepnode_prob,'optimizer':optimizer}
    dic_out = {}

    #1 iteration loop
    if restart=='True':
        dic_hyperparam = {}
        L = len(layer_dims)
        filename = 'L' + str(L) + '_NeuNet_output'
        [dic_hyperparam,dic_out,weights,v,s,t] = np.load(filename+'.npy',allow_pickle=True)
        iter_start = np.max(dic_out['iter'])
        optimizer = dic_hyperparam['optimizer']
        batch_size = dic_hyperparam['batch_size']
        reg_lambd = dic_hyperparam['reg_lambd']
        keepnode_prob = dic_hyperparam['keepnode_prob']
        train_index = dic_hyperparam['train_index']
        dev_index = dic_hyperparam['dev_index']        
    else:
        weights = initialize_weights(layer_dims)
        v, s = initialize_adam(weights)
        iter_start = 1
        t = 0
        
    X_dev = X.iloc[:,dev_index]
    Y_dev = Y.iloc[:,dev_index]

    m = len(train_index)
    batch_num = int(np.ceil(m/batch_size))
    for iter in range(iter_start,iter_num):
        train_index_shuffle = sample(train_index,len(train_index))
        train_batch_index = np.array_split(np.array(train_index_shuffle),batch_num)
        for batch in range(batch_num):
            batch_index = train_batch_index[batch]
            X_train = X.iloc[:,batch_index]
            Y_train = Y.iloc[:,batch_index]
            
        #2 forward propagation
            A_L, layer_mem = forward_propagation(X_train,weights,hidden_activation,output_activation,keepnode_prob)
        #3 backward propagation
            grads = backward_propagation(A_L,Y_train,layer_mem,output_activation,reg_lambd,keepnode_prob)
        #4 update weights
            t = t + 1
            weights, v, s = update_weights(weights,grads,learning_rate,optimizer,v,s,t)
        if iter%10==0:
            print(iter)
        #A compute cost
            cost = cost_function(A_L,Y_train,weights,reg_lambd,output_activation)
        #B compute metrics
            [accuracy,precision,recall,f1] = check_fit(A_L,Y_train,output_activation)
            print('cost=',cost,'acc=',accuracy*100) #,'prec=',precision*100,'rec=',recall*100,'f1=',f1)

            dic_out = {'iter':[],'cost':[],'accuracy':[],'precision':[],'recall':[],'f1':[]}
            dic_out['iter'].append(iter)
            dic_out['cost'].append(cost)
            dic_out['accuracy'].append(accuracy)
            dic_out['precision'].append(precision)
            dic_out['recall'].append(recall)
            dic_out['f1'].append(f1)
            
    L = len(layer_dims)
    filename = 'L' + str(L) + '_NeuNet_output'
    np.save(filename,[dic_hyperparam,dic_out,weights,v,s,t])

    # Evaluate performance on Train and Dev sets
    X_train = X.iloc[:,train_index]
    Y_train = Y.iloc[:,train_index]

    [train_accuracy, train_precision, train_recall, train_f1] = check_fit_dev(X_train,Y_train,weights,hidden_activation,output_activation)
    print('accuracy on train set is ' + str(train_accuracy*100) + '%')
    print('precision on train set is ' + str(train_precision*100) + '%')
    print('recall on train set is ' + str(train_recall*100) + '%')
    print('f1 on train set is ' + str(train_f1))

    [dev_accuracy, dev_precision, dev_recall, dev_f1] = check_fit_dev(X_dev,Y_dev,weights,hidden_activation,output_activation)
    print('accuracy on dev set is ' + str(dev_accuracy*100) + '%')
    print('precision on dev set is ' + str(dev_precision*100) + '%')
    print('recall on dev set is ' + str(dev_recall*100) + '%')
    print('f1 on dev set is ' + str(dev_f1))

######################################################################
def check_fit(A_L,Y,output_activation):

    if output_activation=='sigmoid':
        [accuracy, precision, recall, f1] = check_fit_sigmoid(A_L,Y)
    elif output_activation=='softmax':
        [accuracy, precision, recall, f1] = check_fit_softmax(A_L,Y)
    return accuracy, precision, recall, f1

def check_fit_softmax(A_L,Y):

    Y_targ = np.squeeze(np.array(Y))
    A_pred = np.squeeze(np.array(A_L))

    y_pred = np.argmax(A_pred,axis=0)
    y_hat = np.argmax(Y_targ,axis=0)

    nclass = np.shape(Y)[0]
    accuracy, precision, recall, f1 = compute_accuracy_metrics(y_pred,y_hat,nclass)
    return accuracy, precision, recall, f1

def check_fit_sigmoid(A_L,Y):

    Y_hat = np.squeeze(np.array(Y))
    A_pred = np.squeeze(np.array(A_L))
    Y_pred = np.zeros(np.shape(A_pred))
    Y_pred[np.where(A_pred>=0.5)]=1.0        

    [accuracy, precision, recall, f1] = compute_accuracy_metrics(Y_pred,Y_hat,1)
    return accuracy, precision, recall, f1

def check_fit_dev(X_dev,Y_dev,weights,hidden_activation,output_activation):

    A_L, layer_mem = forward_propagation(X_dev,weights,hidden_activation,output_activation,keepnode_prob=1)  
    [accuracy, precision, recall, f1] = check_fit(A_L,Y_dev,output_activation)
    return accuracy, precision, recall, f1

def compute_accuracy_metrics(Y_pred,Y_hat,nclass):

    accuracy_class=[]; precision_class=[]; recall_class=[]; f1_class=[]    
    for C in range(nclass):
        p = 0; tp = 0; fn= 0; n = 0; tn = 0; fp = 0
        if nclass==1: 
            C=1
        tp=len(np.intersect1d(np.argwhere(Y_pred==C),np.argwhere(Y_hat==C)))
        tn=len(np.intersect1d(np.argwhere(Y_pred!=C),np.argwhere(Y_hat!=C)))
        fp=len(np.intersect1d(np.argwhere(Y_pred==C),np.argwhere(Y_hat!=C)))
        fn=len(np.intersect1d(np.argwhere(Y_pred!=C),np.argwhere(Y_hat==C)))
        p = tp + fn
        n = tn + fp
        try: 
            accuracy = (tp+tn)/(p+n)
        except:
            accuracy = np.nan
        try: 
            precision = tp / (tp + fp)
        except:
            precision = np.nan
        try:
            recall = tp / (tp + fn)
        except:
            recall = np.nan
        try:
            f1 = 2*tp / (2*tp + fp + fn)
        except:
            f1 = np.nan
        accuracy_class.append(accuracy)
        precision_class.append(precision)
        recall_class.append(recall)
        f1_class.append(f1)

    accuracy_class = np.squeeze(np.asarray(accuracy_class))
    precision_class = np.squeeze(np.asarray(precision_class))
    recall_class = np.squeeze(np.asarray(recall_class))
    f1_class = np.squeeze(np.asarray(f1_class))
    return accuracy_class, precision_class, recall_class, f1_class

######################################################################
#1 initialize values
def initialize_weights(layer_dims):

    # layer_dim is the number of units in each layer; using He et al. (2015) initialization
    weights = {}
    L = len(layer_dims)
    for l in range(1, L):
        weights['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2.0/layer_dims[l-1])
        weights['b' + str(l)] = np.zeros((layer_dims[l],1))
    return weights

def initialize_adam(weights) :

    L = len(weights) // 2 # number of layers in the neural networks
    v = {}; s = {}
    # Initialize v, s
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(np.shape(weights["W" + str(l+1)]))
        v["db" + str(l+1)] = np.zeros(np.shape(weights["b" + str(l+1)]))
        s["dW" + str(l+1)] = np.zeros(np.shape(weights["W" + str(l+1)]))
        s["db" + str(l+1)] = np.zeros(np.shape(weights["b" + str(l+1)]))    
    return v, s

#2 forward propagation
def forward_propagation(X,weights,hidden_activation,output_activation,keepnode_prob):

    #layer loop
    layer_mem = []
    A_l = X
    D_l = 1.0
    L = len(weights) // 2 # number of layers in neural network
    for l in range(1, L):
        W_l = weights['W' + str(l)]; b_l = weights['b' + str(l)]; A = A_l; D = D_l
        Z_l = forward(A, W_l, b_l); A_l = activation(Z_l,hidden_activation)
        D_l = np.random.rand(np.shape(A_l)[0],np.shape(A_l)[1]); D_l = (D_l < keepnode_prob).astype(int); A_l = np.multiply(D_l,A_l)/keepnode_prob
        layer_mem.append((A, D, W_l, b_l, Z_l, A_l))
    
    W_L = weights['W' + str(L)]; b_L = weights['b' + str(L)]; A = A_l; D = D_l
    Z_L = forward(A, W_L, b_L); A_L = activation(Z_L,output_activation)
    layer_mem.append((A, D, W_L, b_L, Z_L, A_L))
    return A_L, layer_mem

#3 backward propagation
def backward_propagation(A_L,Y,layer_mem,output_activation,reg_lambd,keepnode_prob):

    #layer loop
    grads = {}
    L = len(layer_mem) # the number of layers
    m = A_L.shape[1]
    # Initializing the backpropagation: Lth layer
    current_layer_mem = layer_mem[L-1]
    dZ_L = deriv_cost(Y,A_L,output_activation)
    dA, dW_L, db_L = backward(dZ_L,current_layer_mem,reg_lambd)
    D = current_layer_mem[1]; dA = np.multiply(D,dA)/keepnode_prob
    grads["dA" + str(L-1)] = dA; grads["dW" + str(L)] = dW_L; grads["db" + str(L)] = db_L

    # Loop for remaining layers
    for l in reversed(range(L-1)):
        dA_l = grads["dA" + str(l+1)];
        current_layer_mem = layer_mem[l]; A_l = current_layer_mem[-1]
        dg_l = deriv_activation(A_l,hidden_activation); dZ_l = dA_l * dg_l
        dA, dW_l, db_l = backward(dZ_l,current_layer_mem,reg_lambd)
        D = current_layer_mem[1]; dA = np.multiply(D,dA)/keepnode_prob
        grads["dA" + str(l)] = dA; grads["dW" + str(l+1)] = dW_l; grads["db" + str(l+1)] = db_l
    return grads

#4 compute cost
def cost_function(Y_pred,Y,weights,reg_lambd,type):

    L = len(weights) // 2
    m = np.shape(Y)[0]
    L2_weights = 0
    for l in range(1,L):
        W_l = weights['W'+str(l)]
        L2_weights = L2_weights + reg_lambd/(2.0)*(np.sum(np.power(W_l,2)))
    cost = compute_cost(Y_pred,Y,type)
    cost = 1.0/m*(cost + L2_weights)
    return cost

#5 update weights
def update_weights(weights, grads, learning_rate, type, v, s, t):

    if type=='adam':
        weights, v, s = update_weights_adam(weights, grads, learning_rate, v, s, t)
    elif type=='gradient_descent':
        weights = update_weights_gd(weights, grads, learning_rate)
    return weights, v, s

######################################################################
#I forward propagation
def forward(A, W_l, b_l):

    Z_l = np.dot(W_l,A) + b_l
    return Z_l

#II backward propagation
def backward(dZ_l,layer_mem,reg_lambd):

    A, D, W_l, b_l, Z_l, A_l = layer_mem
    m = A.shape[1]
    dW_l = 1.0/m*np.dot(dZ_l,A.T) + reg_lambd/m*W_l
    db_l = 1.0/m*np.sum(np.array(dZ_l),axis=1,keepdims=True)
    dA = np.dot(W_l.T,dZ_l)
    return dA, dW_l, db_l

#III compute cost 
def compute_cost(Y_pred,Y,type):

    if type=='sigmoid':
        # 'binary_class'
        cost = -np.sum(np.sum(np.multiply(np.log(Y_pred),Y) + np.multiply(np.log(1-Y_pred),(1-Y))))
    elif type=='softmax':
        # 'multi_class'
        cost  = -np.sum(np.sum(np.multiply(np.log(Y_pred+1e-8),Y)))
    cost = np.squeeze(cost)
    return cost

def deriv_cost(Y,A_L,type):

    if type=='sigmoid':
        dZ_L = np.subtract(A_L,Y )
    elif type=='softmax':
        dZ_L = np.subtract(A_L,Y)
    return dZ_L

def activation(Z,type):

    if type=='sigmoid':
        A = 1 / (1 + np.exp(-Z))
    elif type=='softmax':
        a = np.exp(Z-np.max(Z)); 
        A = np.divide(a,np.sum(a,axis=0,keepdims=True))
    elif type=='relu':
        A = np.maximum(0.01*Z,Z)    # leaky relu
        # A = np.maximum(0,Z)       # pure relu
    elif type=='tanh':
        A = np.tanh(Z)
    return A

def deriv_activation(A,type):

    dg = np.zeros(np.shape(A))
    if type=='sigmoid':
        dg = A*(1 - A)
    elif type=='relu':
        dg[A<0] = 0.01; dg[A>=0] = 1     # leaky relu
#        dg[A<0] = 0; dg[A>=0] = 1     # pure relu 
    elif type=='tanh':
        dg = 1 - np.power(A,2)
    return dg

#IV update weights
def update_weights_gd(weights, grads, learning_rate):

    L = len(weights) // 2 # number of layers in neural network
    #layer loop
    for l in range(L):
        weights["W" + str(l+1)] = weights["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        weights["b" + str(l+1)] = weights["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return weights

def update_weights_adam(weights, grads, learning_rate, v, s, t, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    
    L = len(weights) // 2                 # number of layers in neural network
    v_corrected = {}                      # Initializing 1st moment
    s_corrected = {}                      # Initializing 2nd moment
    
    # Update weights of all layers using adam
    for l in range(L):
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1-beta1)*grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads['db' + str(l+1)]

        # Compute bias-corrected 1st moment
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-beta1**t)

        # Weighted average of squared gradients
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*np.power(grads['dW' + str(l+1)],2)
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*np.power(grads['db' + str(l+1)],2)

        # Compute bias-corrected 2nd moment
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-beta2**t)

        # Update weights
        weights["W" + str(l+1)] = weights["W" + str(l+1)] - learning_rate*v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        weights["b" + str(l+1)] = weights["b" + str(l+1)] - learning_rate*v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
    return weights, v, s
