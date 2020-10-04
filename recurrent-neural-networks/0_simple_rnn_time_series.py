import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

#####################
### Creating Data ###
#####################

plt.figure(figsize=(8,5))

# how many time steps/data pts are in one batch of data
seq_length = 20

# generate evenly spaced data pts
time_steps = np.linspace(0, np.pi, seq_length + 1)
print(f'time_steps : {time_steps}')
print(f'time_steps.shape : {time_steps.shape}')

data = np.sin(time_steps)
print(f'data : {data}')
print(f'data.shape : {data.shape}')
data.resize((seq_length + 1, 1)) # size becomes (seq_length+1, 1), adds an input_size dimension
print(f'data.shape : {data.shape}')

x = data[:-1] # all but the last piece of data
y = data[1:] # all but the first

# display the data
plt.plot(time_steps[1:], x, 'r.', label='input, x') # x
plt.plot(time_steps[1:], y, 'b.', label='target, y') # y

plt.legend(loc='best')
plt.show()

#####################
### Buidling Model ##
#####################

# Step1 Build Model
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        # input_size = The number of expected features in the input x
        # hidden_size = The number of features in the hidden state h
        # num_layers = Number of recurrent layers.
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False

        self.rnn = nn.RNN(input_size, self.hidden_dim, n_layers, batch_first = True)
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x, hidden):

        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)

        batch_size = x.shape[0]

        # get RNN outputs
        # output of shape (seq_len, batch, num_directions * hidden_size)
        #   : tensor containing the output features (h_t) from the last layer of the RNN, for each t.
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        #   : tensor containing the hidden state for t = seq_len.

        r_out, hidden = self.rnn(x, hidden)

        check_r_out = r_out
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)
        check_r_out_resize = r_out

        # get final output
        output = self.fc(r_out)

        return output, hidden, check_r_out, check_r_out_resize

class RNN2(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 16, output_size = 1, num_layers = 1):
        super(RNN2, self).__init__

        self.hidden_dim = hidden_size

        # (batch, seq, feature)
        self.rnn = nn.RNN(input_size, self.hidden_dim, output_size, num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x, hidden):
        # x : (batch, seq_len, input_size)
        seq_length = x.shape[1]
        r_out, hidden  = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)

        return output, hidden

#########################
### Checking Dimension ##
#########################

# test that dimensions are as expected
test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)

# generate evenly spaced, test data pts
time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps)
data.resize((seq_length, 1))
print(f'data.shape : {data.shape}')

test_input = torch.Tensor(data).unsqueeze(0) # give it a batch_size of 1 as first dimension
print('Input size: ', test_input.size())

# test out rnn sizes
test_out, test_h, check_r_out, check_r_out_resize = test_rnn(test_input, None)
print('Output size: ', test_out.size())
print('Hidden state size: ', test_h.size())
print('check_r_out size: ', check_r_out.size())
print('check_r_out_resize size: ', check_r_out_resize.size())

#####################
### Training rnn ####
#####################

# decide on hyperparameters
input_size=1
output_size=1
hidden_dim=32
n_layers=1

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

# Step2 Set loss
criterion = nn.MSELoss()

# Step3 Set optim
optimizer = optim.Adam(rnn.parameters(), lr=0.01)

# Set4 Define Validation -- Skip

# Set5 Define Train
def train(rnn, n_steps, print_every):
    # initialize the hidden state
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):
        # defining the training data
        time_steps = np.linspace(step * np.pi, (step + 1) * np.pi, seq_length + 1)
        data = np.sin(time_steps)
        data.resize((seq_length + 1, 1))  # input_size=1

        x = data[:-1]
        y = data[1:]

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        y_tensor = torch.Tensor(y)

        # outputs from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i % print_every == 0:
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.')  # input
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.')  # predictions
            plt.show()

    return rnn

# train the rnn and monitor results
n_steps = 75
print_every = 15

trained_rnn = train(rnn, n_steps, print_every)
