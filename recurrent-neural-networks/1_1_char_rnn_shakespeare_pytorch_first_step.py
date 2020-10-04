import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tqdm

import matplotlib.pyplot as plt

#######################################
##### Setting functions for vocabs ####
#######################################

import string

all_chars = string.printable
vocab_size = len(all_chars)

# No PADDING
vocab_dict = dict((c, i) for (i,c) in enumerate(all_chars))

# sentence to ids
def str2ints(s, vocab_dict):
    return [vocab_dict[c] for c in s]

# ids to sentence
def int2str(x, vocab_array):
    return "".join([vocab_array[i] for i in x])


###############################
##### Preparing Dataset #######
###############################

# Testing
dir_path = '/Users/philhoonoh/Desktop/scribbling/Udacity/Intro_to_Deep_Learning_with_PyTorch/3_recurrent-neural-networks/data/tinyshakespeare.txt'
# raw_data = open(dir_path, encoding='utf-8').read().strip()
# data = str2ints(raw_data, vocab_dict)
# data = torch.tensor(data, dtype = torch.int64).split(200)
# data[0].shape

class ShakespeareDataset(Dataset):
    def __init__(self, dir_path, vocab_dict, chunk_size = 200):
        super(ShakespeareDataset, self).__init__()

        self.raw_data = open(dir_path, encoding='utf-8').read().strip()
        self.data = str2ints(self.raw_data, vocab_dict)
        self.data = torch.tensor(self.data, dtype = torch.int64).split(chunk_size)

        if len(self.data[-1]) < chunk_size:
            self.data = self.data[:-1]

        self.n_chunks = len(self.data)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        return self.data[idx]

ds = ShakespeareDataset(dir_path, vocab_dict, chunk_size=200)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

#########################
##### Build Model #######
#########################

# Step1 Build Model
class SequenceGenerationNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, hidden_size=50, num_layers=1, dropout=0.2):
        super(SequenceGenerationNet, self).__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout = dropout)
        self.linear = nn.Linear(hidden_size, num_embeddings)

    def forward(self, x, h0 = None):

        # print(f'1. Before Embedding')
        # print(f'x.shape : {x.shape}\n')
        x = self.emb(x)
        # x : (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        # print(f'2. After Embedding')
        # print(f'x.shape : {x.shape}\n')

        x, (last_h, last_c) = self.lstm(x, h0)
        # x : (batch_size, seq_length, embedding_dim) -> (batch_size, seq_length, hidden_size)
        # last_h : (num_layers, batch_size, hidden_size)
        # last_c : (num_layers, batch_size, hidden_size)
        # print(f'3. After LSTM ')
        # print(f'x.shape : {x.shape}')
        # print(f'last_h.shape : {last_h.shape}')
        # print(f'last_c.shape : {last_c.shape}\n')

        x = self.linear(x)
        # x : (batch_size, seq_length, hidden_size) -> (batch_size, seq_length, num_embeddings)
        # print(f'4. After Liner')
        # print(f'x.shape : {x.shape}\n')

        return x, (last_h, last_c)

model = SequenceGenerationNet(num_embeddings = vocab_size, embedding_dim=20, hidden_size=50, num_layers=1, dropout=0.2)

# Step2 Set Loss
criterion = nn.CrossEntropyLoss()

# Step3 Set optim
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Step4 Define Validation (Generate Sequence)
def generate_seq(model, start_phrase="The King said ", length = 200):

    result = []

    start_tensor = torch.tensor(str2ints(start_phrase, vocab_dict), dtype=torch.int64)

    x0 = start_tensor.unsqueeze(0)

    o, h = model(x0)
    # o.shape: torch.Size([1, 14, 100])
    # h[0].shape : torch.Size([1, 1, 50])
    # h[1].shape: torch.Size([1, 1, 50])

    # detach from h, do not back propagate the whole history
    h = [each.data for each in h]

    # Get sampling

    # Method1
    # Sampling From the distribution
    # o[:, -1] == o[:, -1, :]
    # (batch_size, sequence_length, num_embedding) -> (batch_size, last_length, num_embedding) -> (batch_size, num_embedding)
    out_dist = o[:,-1,:].view(-1).exp()
    top_i = torch.multinomial(out_dist, 1)[0]
    top_class = top_i

    # Method2
    # Get the Maximum Probability
    # out_log_ps = torch.log_softmax(o[:,-1,:].view(-1), dim = 0)
    # ps = torch.exp(out_log_ps)
    # top_p, top_class = ps.topk(1, dim=0)

    result.append(top_class)

    for i in range(length):
        inp = torch.tensor([[top_class]], dtype = torch.int64)
        o, h = model(inp, h)
        h = [each.data for each in h]

        out_log_ps = torch.log_softmax(o[:,-1,:].view(-1), dim = 0)
        ps = torch.exp(out_log_ps)
        top_p, top_class = ps.topk(1, dim=0)

        result.append(top_class)

    return start_phrase + int2str(result, all_chars)

# Step5 Define Training
def train_seq(model, loader, criterion, optimizer, epochs = 50, print_every = 50, clip = 5):
    train_losses = []
    h0 = None

    for epoch in range(epochs):
        model.train()
        running_losses = 0
        steps = 0

        for data in tqdm.tqdm(loader):
            steps += 1
            x = data[:, :-1]
            y = data[:, 1:]

            # print(f'x.shape : {x.shape}')
            # print(f'y.shape : {y.shape}')

            # Step 5-1 initilzing optimizer
            optimizer.zero_grad()

            # Step 5-2 get pred
            y_pred, h = model(x, h0 = h0)
            # print(f'y_pred.shape : {y_pred.shape}')
            # print(f'y.shape : {y.shape}')
            # y_pred.shape: torch.Size([32, 199, 100])
            # y.shape: torch.Size([32, 199])

            # Step 5-3 calculate loss
            loss = criterion(y_pred.view(-1, vocab_size), y.reshape(-1))
            running_losses += loss

            # Step 5-4 loss backward & grad_clip
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Step 5-5 optimizer update
            optimizer.step()

            # Step 5-6 detach hidden state from history
            h0 = tuple([each.data for each in h])

            if steps%print_every == 0:
                print(f'Epoch {epoch+1}/{epochs} \t'
                      f'Steps {steps}/{len(loader)} \t'
                      f'Training Loss {running_losses/steps:0.03f}')
        else:
            model.eval()
            with torch.no_grad():
                result = generate_seq(model, start_phrase="The King said ", length = 200)
                print(result)

            train_losses.append(running_losses/len(loader))

            print(f'Epoch {epoch + 1}/{epochs} \t'
                  f'Training Loss {train_losses[-1]:0.03f} \n'
                  f'Result : {result}')
            model.train()

    return model, train_losses

model, train_losses = train_seq(model, loader, criterion, optimizer, epochs = 10, print_every = 50, clip = 5)

# Step6 Plot losses
plt.plot(train_losses, label = 'Training loss')
plt.legend(frameon=False)

#########################
##### Inference #########
#########################

model.eval()
with torch.no_grad():
    result = generate_seq(model, start_phrase="Hello My name is ", length = 200)
    print(result)

