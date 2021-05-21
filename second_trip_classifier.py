import numpy as np
import matplotlib.pyplot as plt
from utils import PolyMask
from netCDF4 import Dataset
import json, configparser, gc, pickle
import tensorflow as tf
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, BatchNormalization, concatenate
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import History
import keras.backend as K



data_dir = './data/'
nfiles = 10                     #number of files to load (set to something small for testing)
input_shape = [768,128]         #input shape for CNN
epoch_size = 1024*5             #training samples per epoch
n_epochs = 100                  #number of training epochs
lr_epochs = 20                  #number of additional epochs at reduced learning rate
batch_size = 16                 #mini-batch size
fields = ['ref','vel','wid']    #final training loss of 0.072 with 4-inputs
class_weight = 0.75             #if classes are skewed this sets the weight of the 1's class in the BCE loss function

#unet++ size:
base_channels = 32
downsamples = 4
growth = 1.5

#get info about the radar fields:
conf = configparser.ConfigParser()
conf.read('radar_vars.ini')
FIELDS = conf._sections
for var in list(FIELDS.keys()):
    FIELDS[var]['clims'] = json.loads(FIELDS[var]['clims'])
    FIELDS[var]['mask_color'] = json.loads(FIELDS[var]['mask_color'])
fields = list(FIELDS.keys())

def load_sweep(fname,elevation):
    ncf = Dataset(fname)
    
    #get the sweep number and starting index and azimuth:
    sweep_idx = np.argmin(np.abs(ncf.variables['fixed_angle'][:]-elevation))
    start_idx = ncf.variables['sweep_start_ray_index'][:][sweep_idx]
    start_azm = ncf.variables['azimuth'][:][start_idx]
    
    #load the sweep:
    sweep = []
    for f in fields:
        data = ncf.variables[FIELDS[f]['netcdf_name']][:][start_idx:start_idx+360,:]
        rng = FIELDS[f]['clims']
        data = 2*(data-rng[0])/(rng[1]-rng[0])-1
        sweep.append(np.clip(data,-1,1))
    sweep = np.array(sweep).transpose((2,1,0))
    sweep = np.roll(sweep,int(start_azm),axis=1)
    return sweep

def load_hand_labeled_data(nfiles):
    #load the masks and the corresponding radar data:
    masks = pickle.load(open('./second_trip.msk','rb'))
    mask_ids = list(masks.keys())
    mask_ids.sort()
    inds = list(np.linspace(0,len(mask_ids)-1,nfiles,dtype='int'))
    mask_ids = [mask_ids[idx] for idx in inds]
    
    raster_masks = []
    radar_data = []
    for mask_id in mask_ids:
        print('Loading case: ' + mask_id)
        fname, elv = mask_id.split('_elv=')
        radar_data.append(load_sweep(data_dir + fname + '.nc',float(elv)))
        raster_masks.append(masks[mask_id].polar_raster().T[:,:,np.newaxis])

    return np.array(radar_data), np.array(raster_masks)

def load_auto_labeled_data(nfiles):
    #this loads some cases and their algorithmically generated classification labels
    elevations = [0.5,1.5,2.6,3.5]
    #unfinished, use idea is to eventually train with a combination of the hand labels, algorithmic labels, and CNN labels as targets
    pass

def batch_generator(data,masks,inputs,targets):
    
    for i in range(targets.shape[0]):
        #get a random sample:
        case_ind = np.random.randint(0,data.shape[0])
        start_azm = np.random.randint(0,data.shape[2]-input_shape[1])
        start_rng = np.random.randint(0,data.shape[1]-input_shape[0])
        sample_data = np.copy(data[case_ind,start_rng:start_rng+input_shape[0],start_azm:start_azm+input_shape[1],:])
        sample_mask = np.copy(masks[case_ind,start_rng:start_rng+input_shape[0],start_azm:start_azm+input_shape[1],:])
        
        #do some augmentation:
        #random flips:
        if np.random.choice([True,False]):
            sample_data = np.flip(sample_data,axis=1)
            sample_mask = np.flip(sample_mask,axis=1)
            
        #salt and pepper noise for inputs:
        snp = np.double(np.random.uniform(0,1,size=sample_data.shape)>0.95)*np.random.uniform(-1,1,sample_data.shape)
        sample_data = np.clip(-1,1,sample_data+snp)
        
        #randomly flip mask pixels:
        flips = np.double(np.random.uniform(0,1,size=sample_mask.shape)>0.995)
        sample_mask = (-(sample_mask*2-1)*(flips*2-1)+1)/2
        
        #soft labels:
        sample_mask = sample_mask*0.9 + np.random.uniform(0,0.1,sample_mask.shape)
        
        #random distortion of input field intensity:
        for j in range(sample_data.shape[-1]):
            distortion = np.random.normal(0,0.1)
            sample_data[:,:,j] = np.clip(-1,1,sample_data[:,:,j] + distortion)
        
        inputs[i,:,:,:] = sample_data
        targets[i,:,:,:] = sample_mask

def weighted_binary_crossentropy(y_true,y_pred):
    bce = K.binary_crossentropy(y_true,y_pred)
    wbce = 0.75*y_true*bce + 0.25*(1-y_true)*bce
    return tf.reduce_mean(wbce)
    
#defines the convolutions done in the unet
def conv(x,channels,filter_size=3):
    x = Conv2D(channels,(filter_size,filter_size),padding='same',activation='linear')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    return x

#creates unet ++ style network
def unetpp(INPUT_SIZE,base_channels=32,levels=4,growth=1.5):
    #defines the unet++
    xin = Input(INPUT_SIZE)
    net = []
    for lev in range(levels):
        if lev == 0:
            net.append([concatenate([xin,conv(xin,base_channels)])])
        else:
            net_layer = []
            for proc in range(lev+1):
                inputs = []
                if proc < lev:
                    inputs.append(MaxPooling2D((2,2))(net[lev-1][proc]))
                if proc > 0:
                    inputs.append(UpSampling2D((2,2))(net_layer[proc-1]))
                    inputs.append(net[lev-1][proc-1])
                if len(inputs)>1:
                    inputs = concatenate(inputs)
                else:
                    inputs = inputs[0]
                output = conv(inputs,int(base_channels*growth**(lev-proc)))
                if proc>0:
                    output = concatenate([output,net[lev-1][proc-1]])
                net_layer.append(output)
            net.append(net_layer)
    x = conv(net[-1][-1],base_channels*levels)
    #combine input with generated output:
    xout = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)
    cnn = Model(xin,xout)
    return cnn

def train_cnn(cnn,data,labels):
    inputs = np.zeros((epoch_size,input_shape[0],input_shape[1],len(fields)))
    targets = np.zeros((epoch_size,input_shape[0],input_shape[1],1))
    cnn.compile(optimizer=Adam(learning_rate = 0.0005),loss=weighted_binary_crossentropy)
    loss = []
    for i in range(n_epochs+lr_epochs):
        print('Epoch: ' + str(i))
        batch_generator(data,labels,inputs,targets)
        hist = cnn.fit(inputs,targets,batch_size = batch_size)
        loss.append(hist.history['loss'][0])
        np.save('training_loss.npy',loss)
        if i == n_epochs:
            K.set_value(cnn.optimizer.lr,0.1*K.eval(cnn.optimizer.lr))
        gc.collect()
    cnn.save('model')
    
    
if __name__=='__main__':
    data,labels = load_hand_labeled_data(nfiles)
    cnn = unetpp(input_shape + [len(fields)], base_channels, downsamples, growth)
    train_cnn(cnn,data,labels)