import itertools
import datetime
import pandas as pd
import numpy as np
import requests
import math
import os
import warnings
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import horovod.torch as hvd



def read_file_from_aws():
  url = "https://medium-post-data.s3.amazonaws.com/data_es.csv"
  response = requests.get(url, stream = True)

  text_file = open("data_es.csv","wb")
  for chunk in response.iter_content(chunk_size=1024):
    text_file.write(chunk)


read_file_from_aws()

x_cols = ['close', 'ask', 'bid', 'md_0_ask', 'md_0_bid',
       'md_1_ask', 'md_1_bid', 'md_2_ask', 'md_2_bid', 'md_3_ask', 'md_3_bid',
       'md_4_ask', 'md_4_bid', 'md_5_ask', 'md_5_bid', 'md_6_ask', 'md_6_bid',
       'md_7_ask', 'md_7_bid', 'md_8_ask', 'md_8_bid', 'md_9_ask', 'md_9_bid']

y_cols = ['close']

def reshape_and_scale_data_for_training(data, window_size, x_cols, y_cols, y_len = 1, scale=True, scaler=MinMaxScaler, test_size=0.2, backend='keras'):
  """
  given an instance of  ```pandas.DataFrame``` and a window size, this reshapes the data and scales it for training. You can
  pass in the column names of your x's and y's.
  """
  
  def cols_to_indices(data, cols):
    return [list(data.columns).index(i) for i in cols]
  
  def get_new_pos(i, *indices):
    originals = sorted(list(set([i for i in itertools.chain(*indices)])))
    return originals.index(i)

  _l = len(data)
  _x = []
  _y = []

  x_indices = cols_to_indices(data, x_cols)
  y_indices = cols_to_indices(data, y_cols)

  s = None
  if scale:
    s = scaler()
    data = data.iloc[:, list(set(x_indices + y_indices))]
    data = s.fit_transform(data)
  else:
    # convert this to an np array to conform to the index access pattern below
    data = np.array(data.iloc[:, list(set(x_indices + y_indices))])

  xs = np.array(data[:, [get_new_pos(i, x_indices, y_indices) for i in x_indices]])
  ys = np.array(data[:, [get_new_pos(i, x_indices,y_indices) for i in y_indices]])

  for i in range(0, _l - window_size):
    _x.append(xs[i:i + window_size])
    _y.append(ys[i + window_size:((i + window_size) + y_len)])

  
  x_train, x_test, y_train, y_test= train_test_split(_x, _y, test_size=test_size, shuffle=False)
  if backend == 'keras':
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), s 
  if backend == 'torch':
    return torch.Tensor(x_train), torch.Tensor(x_test), torch.Tensor(y_train), torch.Tensor(y_test), s


class TimeSeriesDataSet(Dataset):
  """
  This is a custom dataset class. It can get more complex than this, but simplified so you can understand what's happening here without
  getting bogged down by the preprocessing
  """
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    _x = self.X[index]
    _y = self.Y[index]

    return _x, _y



class LSTM(nn.Module):
  """implements an lstm - a single/multilayer uni/bi directional lstm"""
  def __init__(self, n_features, window_size, 
               output_size, h_size, n_layers=1, 
               bidirectional=False, device=torch.device('cpu'), initializers=[]):
    super().__init__()
    self.n_features = n_features
    self.window_size = window_size
    self.output_size = output_size
    self.h_size = h_size
    self.n_layers = n_layers
    self.directions = 2 if bidirectional else 1
    self.device = device

    self.lstm = nn.LSTM(input_size=n_features, hidden_size=h_size, 
                        num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
    self.hidden = None
    
    self.linear = nn.Linear(self.h_size * self.directions, self.h_size)
    self.linear2 = nn.Linear(self.h_size, 64)
    self.linear3 = nn.Linear(64, 1)

    self.layers =  [self.lstm, self.linear, self.linear2, self.linear3]
    self.initializers = initializers
    
    self._initialize_all_layers()
   
    self.tensors = {
        "float": torch.cuda.FloatTensor if str(self.device) == "cuda" else torch.FloatTensor,
        "long": torch.cuda.LongTensor if str(self.device) == "cuda" else torch.LongTensor
    }

  def _initialize_all_layers(self):
    """
    
    If the user provides initializers, initializes each layer[i] with initializer[i]
    If no initializers provided, moves the layers to the device specified

    """
    if all([self.initializers, len(self.initializers) != self.layers, 
            len(self.initializers) == 1]):  
      warnings.warn("only one initializer: {} was provided for {} layers, the initializer will be used for all layers"\
                    .format(len(self.layers)))
      if len(self.initializers) == 1:
        [self._initialize_layer(self.initializers[0],x) for x in self.layers]
      else:
        [self._initialize_layer(self.initializers[i],x) for i, x in enumerate(self.layers)]
    
    elif all([self.initializers, 
              len(self.initializers) != self.layers]):
      raise Exception("{} initializers were provided for {} layers, need to provide an initializer for each layer"\
                      .format(len(self.initializers), len(self.layers)))
    else:
      # uses default initialization, but moves layer to device
      [self._initialize_layer(None, x) for x in self.layers]

  def _initialize_layer(self, initializer, layer):
    if initializer:
      pass
    layer.to(self.device)

  def _make_tensor(self, tensor_type, *args, **kwargs):
    """
     returns a tensor appropriate for the device
     
     :param tensor_type: supported ['long', 'float'] 
     :returns: a torch.Tensor
    """
    t = self.tensors[tensor_type](*args,**kwargs).to(self.device)
    return t

  def init_hidden(self, batch_size):
    
    hidden_a  = torch.randn(self.n_layers * self.directions,
                            batch_size ,self.h_size).to(self.device)
    hidden_b = torch.randn(self.n_layers * self.directions, 
                           batch_size,self.h_size).to(self.device)
    
    hidden_a = Variable(hidden_a)
    hidden_b = Variable(hidden_b)

    return (hidden_a, hidden_b) 

  def forward(self, input):
    batch_size =  list(input.size())[0]
    self.hidden = self.init_hidden(batch_size)
    
    lstm_output, self.hidden = self.lstm(input, self.hidden)  
    last_hidden_states = torch.index_select(lstm_output, 1,  index=self._make_tensor("long", ([self.window_size-1])))
    
    
    predictions = self.linear3(
        self.linear2(
            self.linear(
                last_hidden_states
                )
            )
        )
    return predictions

if __name__ == "__main__":

  # intialize horovod
  hvd.init()

  # to handle dynamically updating GPUs
  # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  gpus = [gpu for gpu in GPUtil.getGPUs()]
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g.id) for g in gpus])


  epochs = 100
  # Horovod: adjust number of epochs based on number of GPUs.
  epochs = int(math.ceil(epochs / hvd.size()))
  window_length = 10

  # Pin GPU to be used to process local rank (one GPU per process)
  if torch.cuda.is_available():
      torch.cuda.set_device(hvd.local_rank())

  _DEVICE = torch.device("cuda:{}".format(str(torch.cuda.current_device())))

  print("this process has been distributed to the following devices: {}"\
      .format(["{}, device_id: cuda:{}"\
               .format(gpu.name, gpu.id) for gpu in GPUtil.getGPUs()]), flush=True)

  print(f"this process is using device - {_DEVICE}", flsh=True)



  df = pd.read_csv("data_es.csv")
  x_train, x_test, y_train, y_test, scaler =  reshape_and_scale_data_for_training(df, window_length, x_cols, y_cols, y_len=1, scale=True,backend='torch')

  train_sampler = torch.utils.data.distributed.DistributedSampler(
    TimeSeriesDataSet(x_train, y_train), num_replicas=hvd.size(), rank=hvd.rank())

  train_loader = DataLoader(TimeSeriesDataSet(x_train, y_train), batch_size=32, pin_memory=True, num_workers=4, sampler=train_sampler)
  test_loader = DataLoader(TimeSeriesDataSet(x_test, y_test), batch_size=len(x_test), shuffle=True, pin_memory=True, num_workers=4)




  model = LSTM(n_features=23, window_size=window_length, output_size=1, h_size=256, device=_DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  

  model.to(_DEVICE)

  loss_fn = nn.MSELoss(reduce=True, reduction='mean')
  iter = 0
  train_times = []

  hvd.broadcast_parameters(model.state_dict(), root_rank=0)

  for epoch in range(epochs):
    start_time = datetime.datetime.now()
    for i, data in enumerate(train_loader):
      # move x and y to the gpu
      inputs, labels = data[0].to(_DEVICE), data[1].to(_DEVICE)
      pred = model(inputs)
      # Calculate Loss: softmax --> cross entropy loss
      loss = loss_fn(pred, labels)
      # Getting gradients w.r.t. parameters
      loss.backward()

      # Updating parameters
      optimizer.step()
      optimizer.zero_grad()
      iter +=1
    for i, test_data in enumerate(test_loader):
      # move x and y to the gpu
      test_inputs, test_labels = test_data[0].to(_DEVICE), test_data[1].to(_DEVICE)
      test_pred = model(test_inputs)
      test_loss = loss_fn(test_pred, test_labels)
      print('Epoch: {}, Iteration: {}. train_loss: {},  valid_loss: {}'.format(epoch, iter, loss, test_loss))
    end_time = datetime.datetime.now()
    time = (end_time - start_time).total_seconds()
    train_times.append(time)
    print("time taken: {}".format(time))
