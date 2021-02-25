import os, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import datasets, models, transforms, utils
import PIL
#####################################################################
class ConvLayerMulti(nn.Module):

    def __init__(self, layers_dim,keepnode_prob):
        super(ConvLayerMulti, self).__init__()
        self.conv1  = nn.Conv2d(layers_dim['input'][0], layers_dim['conv1'][3], layers_dim['conv1'][0], layers_dim['conv1'][1], layers_dim['conv1'][2])
        self.norm1  = nn.BatchNorm2d(layers_dim['conv1'][3])
        self.dout1  = nn.Dropout2d(1-keepnode_prob)
        self.pool1  = nn.MaxPool2d(layers_dim['pool1'][0],layers_dim['pool1'][1])
        self.conv2  = nn.Conv2d(layers_dim['conv1'][3], layers_dim['conv2'][3], layers_dim['conv2'][0], layers_dim['conv2'][1], layers_dim['conv2'][2])
        self.norm2  = nn.BatchNorm2d(layers_dim['conv2'][3])
        self.dout2  = nn.Dropout2d(1-keepnode_prob)
        self.pool2  = nn.MaxPool2d(layers_dim['pool2'][0],layers_dim['pool2'][1])
        self.flat3  = nn.Flatten()
        self.fc3    = nn.Linear(layers_dim['fc3'], layers_dim['output'])
        self.dout3  = nn.Dropout(1-keepnode_prob)

    # Defining the forward pass    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
#        x = self.dout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
#        x = self.dout2(x)
        x = self.pool2(x)

        x = self.flat3(x)
        x = F.log_softmax(self.fc3(x))
        return x

class FullConnectMulti(nn.Module):
    def __init__(self, layer_dims,keepnode_prob):
        super(FullConnectMulti, self).__init__()
        self.fc1 = nn.Linear(layer_dims[0], layer_dims[1])
        self.fc2 = nn.Linear(layer_dims[1], layer_dims[2])
        self.fc3 = nn.Linear(layer_dims[2], layer_dims[-1])
        self.dropout = nn.Dropout(p=1-keepnode_prob)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc3(x))
        return x 

class FullConnectBinary(nn.Module):
    def __init__(self, layer_dims,keepnode_prob):
        super(FullConnectBinary, self).__init__()
        self.fc1 = nn.Linear(layer_dims[0], layer_dims[1])
        self.fc2 = nn.Linear(layer_dims[1], layer_dims[2])
        self.fc3 = nn.Linear(layer_dims[2], layer_dims[-1])
        self.dropout = nn.Dropout(p=1-keepnode_prob)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        return x 

def master_nn_pytorch(train,dev,class_name,hidden_activation,output_activation,learning_rate,layer_dims,batch_size,iter_num,optimizer,reg_lambd=0.0,keepnode_prob=1.0,restart='False'):

    L = len(layer_dims)
    filename = 'L' + str(L-1) + '_NeuNet_pytorch_output'
###    filename = output_activation

    ###############################################
    # 0. load data
    train_loader = torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev,batch_size=batch_size,shuffle=False)

    # I. load model image_softmax
    model = define_model(class_name,output_activation,layer_dims,keepnode_prob)

    if restart=='True':
        model.load_state_dict(torch.load(filename))

    # II. define optimizer for gradient descent and associated loss function
    if optimizer=='gradient_descent':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # II. 
    if ('vgg' in output_activation) or ('resnet' in output_activation): 
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.NLLLoss()

    # III. train model
    t = 0
    for iter in range(iter_num):
        for input, label in train_loader:
            
            # Forward propagation
            output = model(input.float())
            cost = criterion(output,label)

            # backward propagation
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            t = t + 1
            if t%1==0:
                print('cost=',cost.item())

    torch.save(model.state_dict(), filename)  
    np.save(filename, layer_dims)

    ##############################################    
    # 1. Assess performance on train and dev sets
#    accuracy_score(train_y, predictions)
    with torch.no_grad():
        tptn = 0
        pn = 0

        for input, label in dev_loader:
            output = model(input.float())
            _, Out = torch.max(output.data, 1)
            pn += label.size(0)
            tptn += ( Out == label).sum().item()
        print('Accuracy on dev set: {} %'.format(100 * tptn / pn))

#        for input, label in train_loader:
#            output = model(input.float())
#            _, Out = torch.max(output.data, 1)
#            pn += label.size(0)
#            tptn += ( Out == label).sum().item()
#        print('Accuracy on train set: {} %'.format(100 * tptn / pn))

def define_model(output_activation,class_name,layer_dims,keepnode_prob):
    if output_activation=='custom_fc':
        model = FullConnectMulti(layer_dims,keepnode_prob)
    elif output_activation=='custom_convnet':
        model = ConvLayerMulti(layer_dims,keepnode_prob)
    elif output_activation=='resnet_finetune':
        model = models.resnet18(pretrained=True)       
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_name))
    elif output_activation=='resnet_featextr':
        model = models.resnet18(pretrained=True)       
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_name))
    elif output_activation=='vgg16_featextr':
        model = models.vgg16(pretrained=True)       
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_name))
    return model

def predict_image(output_activation,train,dev,class_name,layer_dims):

###
    L = len(layer_dims)-1
    filename = 'L' + str(L) + '_NeuNet_pytorch_output'
    print(filename)
###    filename = output_activation

    layer_dims = np.load(filename+'.npy',allow_pickle=True).item()
    class_name = layer_dims['class']
    model = define_model(output_activation,class_name,layer_dims,1)
#    model = ConvLayerMulti(layer_dims,1.0)
#    model = models.resnet18(pretrained=False)
#    num_ftrs = model.fc.in_features
#    model.fc = nn.Linear(num_ftrs, len(class_name))

    model.load_state_dict(torch.load(filename))

    # flash outsider image
    in_img_path = '/Users/xl332/Desktop/NinaPics/'
    filename = random.choice(os.listdir(in_img_path))
    print(filename)
    img_name = os.path.join(in_img_path, filename)
#    img_name = 'IMG-20181004-WA0002.jpg'
    img_label = 'unknown'   
    return_class(model,class_name,img_name,img_label,'Nina')

    # flash insider image
    path = '/Users/xl332/Desktop/DataProjects/NeuNet/DogBreed/'
    in_img_path = '/Users/xl332/Desktop/DataProjects/NeuNet/DogBreed/train/train_set/'
    labels = pd.read_csv(path+'labels.csv')
    rand_label = random.randrange(0,np.shape(labels)[0])
    img_name = in_img_path+labels.iloc[rand_label,0]+'.jpg'
    img_label = labels.iloc[rand_label,1]
    return_class(model,class_name,img_name,img_label,'test')

def return_class(model,class_name,img_name,img_label,save_name):
   
    image = PIL.Image.open(img_name)
    transform = transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop((224,224)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    image_tensor = transform(image)[:3,:,:].unsqueeze_(0)
    model.eval()
    idx = torch.argmax(model(image_tensor))
    class_index = class_name[idx]

    fig = plt.figure()    
    plt.imshow(transforms.ToPILImage()(transforms.ToTensor()(image)),interpolation="None")
    plt.title('pred:'+class_index+'; true:'+img_label)
    fig.savefig(save_name+'.png')
    plt.close()

#############
# Saving and restarting from last iter point

#        X_train = X.iloc[:,train_index]
#        Y_train = Y.iloc[:,train_index]
#        X_dev = X.iloc[:,dev_index]
#        Y_dev = Y.iloc[:,dev_index]
#        print(np.shape(X))
#        print(np.shape(Y))
#        train = torch.utils.data.TensorDataset(torch.tensor(np.array(X_train.T)), torch.tensor(np.array(Y_train.T)))
#        dev = torch.utils.data.TensorDataset(torch.tensor(np.array(X_dev.T)), torch.tensor(np.array(Y_dev.T)))

#    dic_hyperparam = {'hidden_activation':hidden_activation,'output_activation':output_activation,'learning_rate':learning_rate,'layer_dims':layer_dims,'train_index':train_index,'dev_index':dev_index,'batch_size':batch_size, 'reg_lambd':reg_lambd,'keepnode_prob':keepnode_prob,'optimizer':optimizer}
#    dic_out = {}

    #1. Initialize
#    if restart=='True':
#        dic_hyperparam = {}
#        L = len(layer_dims)
#        filename = 'L' + str(L) + '_NeuNet_pytorch_output'
#        [dic_hyperparam,dic_out,weights,t] = np.load(filename+'.npy',allow_pickle=True)
#        iter_start = np.max(dic_out['iter'])
#        optimizer = dic_hyperparam['optimizer']
#        batch_size = dic_hyperparam['batch_size']
#        reg_lambd = dic_hyperparam['reg_lambd']
#        keepnode_prob = dic_hyperparam['keepnode_prob']
#        train_index = dic_hyperparam['train_index']
#        dev_index = dic_hyperparam['dev_index']        
#    else:
#        iter_start = 0
#        t = 0


#    image_tensor = transform(image).float()
#    image_tensor = image_tensor.unsqueeze_(0)
#    output = model(image_tensor)
#    image_softmax = np.exp(output.data.numpy())
#    image_softmax = output.data.numpy()
#    index = image_softmax.argmax()
#    print('index:',index)
#    class_index = class_name[index]
