# For transformer 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Conv1D,Flatten,MaxPooling1D,Input,AveragePooling1D,Concatenate,UpSampling1D,UpSampling2D
from keras.models import Model 
import numpy as np 
from keras import backend as K


import os
import tempfile

# pylint: disable=g-bad-import-order
from six.moves import xrange  # pylint: disable=redefined-builtin
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.transformer import compute_bleu
from official.transformer import translate
from official.transformer.model import model_params
from official.transformer.model import transformer
from official.transformer.utils import dataset
from official.transformer.utils import metrics
from official.transformer.utils import schedule
from official.transformer.utils import tokenizer
from official.utils.accelerator import tpu as tpu_util
from official.utils.export import export
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers

PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


DEFAULT_TRAIN_EPOCHS = 10
INF = int(1e9)
BLEU_DIR = "bleu"

# Dictionary containing tensors that are logged by the logging hooks. Each item
# maps a string to the tensor name.
TENSORS_TO_LOG = {
    "learning_rate": "model/get_train_op/learning_rate/learning_rate",
    "cross_entropy_loss": "model/cross_entropy"}

PARAMS_MAP["vocab_size"] = 10
PARAMS_MAP["hidden_size"] = 0
PARAMS_MAP["num_hidden_layers"] = 6
PARAMS_MAP["tpu"] = False
PARAMS_MAP["num_heads"] = 6
PARAMS_MAP["attention_dropout"] = 0
PARAMS_MAP["filter_size"] = 10
PARAMS_MAP["relu_dropout"] = 0
PARAMS_MAP["allow_ffn_pad"] = True
PARAMS_MAP["layer_postprocess_dropout"] = 0

model = None 
sess = None 
init = None 

tf.set_random_seed(10)

def buildUpsampleCNNModel(input_X):

    input_X = Input(shape=(input_X.shape[1],input_X.shape[2]),name= 'input_X')

    input_X =  K.transpose(input_X,(0,2,1))

    upsample_1 = UpSampling1D(size = 2)(input_X)


    conv_1  = Conv1D(4800,4,strides=1,padding='same',name = 'conv_1',kernel_initializer = "glorot_uniform")(upsample_1)

    return Model(inputs = input_X, outputs = upsample_1 )


def buildCNNModel(input_X,initializer):
    
    input_X = Input(shape=(input_X.shape[1],input_X.shape[2]),name= 'input_X')

    conv_1  = Conv1D(1800,5,strides = 1, padding ='same',name = 'conv_1',kernel_initializer = initializer)(input_X)

    conv_2  = Conv1D(4800,5,strides = 1, padding ='same',name = 'conv_2',kernel_initializer = initializer)(conv_1)

    max_pool = MaxPooling1D(pool_size = int(input_X.shape[1]))(conv_2)

    return Model(inputs = input_X, outputs = max_pool )
    
def buildDialatedCNNModel(input_X,initializer):

    input_X = Input(shape=(input_X.shape[1],input_X.shape[2]),name= 'input_X')

    conv_1  = Conv1D(1800,4,padding='same',dilation_rate=1, name = 'conv_1',kernel_initializer =initializer)(input_X)

    conv_2  = Conv1D(4800,4,padding ='same',dilation_rate=1, name = 'conv_2',kernel_initializer =initializer)(conv_1)

    max_pool = MaxPooling1D(pool_size =int(input_X.shape[1]) )(conv_2)

    return Model(inputs = input_X, outputs = max_pool )

def getCNNOutput(para,batch):
    global model 
    if model == None:
        model = buildCNNModel(np.ones([1,para['max_length'],batch.shape[2]]),'glorot_uniform')
    output = model.predict(batch)
    
    return output.reshape(output.shape[0],output.shape[2])

def getDiaCNNOutput(para,batch):
    global model 
    if model == None:
        model = buildDialatedCNNModel(np.ones([1,para['max_length'],batch.shape[2]]),'glorot_uniform')
    output = model.predict(batch)
    return  output.reshape(output.shape[0],output.shape[2])


def POI_Pooling(data):
    embSize = data[0].shape[1]
    returnData = []
    if embSize== 300:
        print("Emb size is 300")

        # Resize length to 16
        length = 16
        for (i,sent) in enumerate(data):
            sent[~np.all(sent == 0, axis=1)]
            if sent.shape[0] <= length:
                resize =  sent.flatten() 
            else: 
                resize = []
                cutPct = (sent.shape[0] - 1 )/length
                for j in range(length):
                    resize.append(sent[int(j*cutPct):int((j+1)*cutPct)].max(axis=0).tolist())
                resize =  np.array(resize).flatten()
            resize = np.pad(resize,pad_width=(0,embSize*length -len(resize)),mode = 'constant')    
            returnData.append(resize)
    else:
        length = int(5000/embSize)
        for (i,sent) in enumerate(data):
            sent = sent[~np.all(sent == 0, axis=1)]
            if sent.shape[0] <= length:
                resize =  sent.flatten() 
            else: 
                resize = []
                cutPct = (sent.shape[0] -1 )/length
                for j in range(length):
                    resize.append(sent[int(j*cutPct):int((j+1)*cutPct)].max(axis=0).tolist())
                resize =  np.array(resize).flatten()
                #print(resize.shape)
            resize = np.pad(resize,pad_width=(0,embSize*length -len(resize)),mode = 'constant')    
            returnData.append(resize)

    return np.array(returnData)
        
def getTransformer(para,batch):

    global model 
    global sess
    global init 

    EmbedType = para['embed']
    if EmbedType == 'Glove':
        #print("GLOVE")
        PARAMS_MAP["hidden_size"] = 300
        PARAMS_MAP["num_heads"] = 6
    if EmbedType == 'GPT2' or EmbedType == 'BERT':
        #print("BERT OR GPT2")
        PARAMS_MAP["hidden_size"] = 768
        PARAMS_MAP["num_heads"] = 6
    if EmbedType == 'ELMO':
        PARAMS_MAP["hidden_size"] = 1024
        PARAMS_MAP["num_heads"] = 8
    if EmbedType == 'Gensen':
        PARAMS_MAP["hidden_size"] = 4096
        PARAMS_MAP["num_heads"] = 16

    if PARAMS_MAP["hidden_size"] == 0:
        raise("Can't not recognize the embed")
    
    # if model == None:
    #print("Transformer")
    # if model == None:
    attention_bias = np.zeros([len(batch),1,1,len(batch[0])])
    
    if model == None:
        tf.reset_default_graph()
        model = transformer.Transformer(PARAMS_MAP, True)
        output = model.encode(batch,attention_bias)
        init = tf.global_variables_initializer()
        sess = tf.Session() 
        sess.run(init)    

    output = model.encode(batch,attention_bias)  
    x = sess.run(output) 
    return x 


    
