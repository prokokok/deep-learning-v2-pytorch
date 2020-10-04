# Implementing Pytorch First Step

import torch
import glob
import pathlib
import re
import numpy as np

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tqdm
import matplotlib.pyplot as plt

###############################
##### Setting functions
#### Text -> ID
#### ID -> Tensor
###############################

remove_marks_regex = re.compile("[,\.\(\)\[\]\*:;]|<.*?>")
shift_marks_regex = re.compile("([?!])")


def text2ids(text, vocab_dict):
    # !? 이외의 기호 삭제
    text = remove_marks_regex.sub("", text)
    # !?와 단어 사이에 공백 삽입
    text = shift_marks_regex.sub(r" \1 ", text)
    tokens = text.split()

    # unkown token index = 0
    return [vocab_dict.get(token, 0) for token in tokens]


def list2tensor(token_idxes, max_len=100, padding=True):
    if len(token_idxes) > max_len:
        token_idxes = token_idxes[:max_len]
    n_tokens = len(token_idxes)
    if padding:
        token_idxes = token_idxes + [0] * (max_len - len(token_idxes))
    return torch.tensor(token_idxes, dtype=torch.int64), n_tokens


# Testing Functions
# test_text = """Wow, another Kevin Costner hero movie. Postman, Tin Cup, Waterworld, Bodyguard, Wyatt Earp, Robin Hood, even that baseball movie. Seems like he """
#
# vocab_path = pathlib.Path('/Users/philhoonoh/Desktop/scribbling/Udacity/Intro_to_Deep_Learning_with_PyTorch/3_recurrent-neural-networks/data/aclImdb copy/imdb.vocab')
# vocab_array = vocab_path.open(encoding='utf-8').read().strip().splitlines()
# vocab_dict = dict((w, i + 1) for (i,w) in enumerate(vocab_array))
#
# token_index = text2ids(test_text, vocab_dict)
# list2_tensor = list2tensor(token_index, max_len=100, padding=True)

###############################
##### Preparing Dataset #######
###############################

class IMDBDataset(Dataset):
    def __init__(self, dir_path, train = True, max_len = 100, padding = True):
        self.max_len = max_len
        self.padding = padding

        path = pathlib.Path(dir_path)
        vocab_path = path.joinpath("imdb.vocab")

        # Constructing vocab_dict
        self.vocab_array = vocab_path.open(encoding='utf-8').read().strip().splitlines()
        self.vocab_dict = dict((w, i + 1) for (i,w) in enumerate(self.vocab_array))

        if train:
            self.target_path = path.joinpath("train")
        else:
            self.target_path = path.joinpath("test")

        pos_files = sorted(glob.glob(str(self.target_path.joinpath("pos/*.txt"))))
        neg_files = sorted(glob.glob(str(self.target_path.joinpath("neg/*.txt"))))

        self.labeled_files = list(zip([0]*len(neg_files), neg_files)) + list(zip([1]*len(pos_files), pos_files))

    @property
    def vocab_size(self):
        return len(self.vocab_array)

    def __len__(self):
        return len(self.labeled_files)

    def __getitem__(self, idx):
        label, f = self.labeled_files[idx]

        data = open(f, encoding='utf-8').read().lower().strip()

        data = text2ids(data, self.vocab_dict)
        data, n_tokens = list2tensor(data, self.max_len, self.padding)
        return data, label, n_tokens

train_data = IMDBDataset('/Users/philhoonoh/Desktop/scribbling/Udacity'
                         '/Intro_to_Deep_Learning_with_PyTorch/3_recurrent-neural-networks/data/aclImdb')

test_data = IMDBDataset('/Users/philhoonoh/Desktop/scribbling/Udacity'
                         '/Intro_to_Deep_Learning_with_PyTorch/3_recurrent-neural-networks/data/aclImdb', train = False)

train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 32, shuffle = True)

#########################
##### Build Model #######
#########################

## Summary of Shape Transformations ##
##----------------------------------##
# https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
# (batch_size X max_seq_len X embedding_dim) --> Sort by seqlen ---> (batch_size X max_seq_len X embedding_dim)
# (batch_size X max_seq_len X embedding_dim) --->      Pack     ---> (batch_sum_seq_len X embedding_dim)
# (batch_sum_seq_len X embedding_dim)        --->      LSTM     ---> (batch_sum_seq_len X hidden_dim)
# (batch_sum_seq_len X hidden_dim)           --->    UnPack     ---> (batch_size X max_seq_len X hidden_dim)

# Step1 Build Model
class SequenceTaggingNetPackedSequence(nn.Module):
    def __init__(self, num_embedding, embedding_dim=50, hidden_size=50, num_layers = 1, dropout = 0.2):
        super(SequenceTaggingNetPackedSequence, self).__init__()
        self.emb = nn.Embedding(num_embedding, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout = dropout)

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h0 = None, l = None):

        # print(f'1. Before Embedding')
        # print(f'x.shape : {x.shape}\n')
        x = self.emb(x)
        # x : (batch, seq_len) -> (batch, seq_len, embedding_dim)
        # print(f'2. After Embedding')
        # print(f'x.shape : {x.shape}\n')

        if l is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True)

        # print(f'3. After pack_padded_sequence')
        # print(f'x.data.shape : {x.data.shape}')
        # print(f'x.batch_sizes : {x.batch_sizes}\n')

        x, (h_last, c_last) = self.lstm(x, h0)
        # (seq_len, batch, num_directions * hidden_size)
        # x : (batch, seq_len) -> (batch, seq_len, hidden_size)
        # h : (batch, seq_len) -> (batch, hidden_size)
        # print(f'4. After LSTM pack_padded_sequence')
        # print(f'x.data.shape : {x.data.shape}')
        # print(f'h_last.shape : {h_last.shape}')
        # print(f'c_last.shape : {c_last.shape}\n')

        # output, input_sizes = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        if l is not None:
            x = h_last[-1]
        else:
            x = x[:,-1,:]
        # x : (batch, seq_len, hidden_size) -> (batch, hidden_size)
        # print(f'5. After Unpacking processing')
        # print(f'x.shape : {x.shape}\n')

        x = self.linear(x)
        # x : (batch, hidden_size) -> (batch, 1)

        # print(f'6. After Liner')
        # print(f'x.shape : {x.shape}\n')

        x = x.squeeze()
        # x : (batch, 1, 1) -> (batch,)
        # print(f'7. After Sqeeze')
        # print(f'x.shape : {x.shape}\n')

        return x

net = SequenceTaggingNetPackedSequence(train_data.vocab_size + 1, num_layers = 1)

# Step2 Set Loss Function
criterion = nn.BCEWithLogitsLoss()

# Step3 Set Optimizer
optimizer = optim.Adam(net.parameters())

# Step4 Define Validation
def validation(model, criterion, test_loader):

    test_loss = 0
    test_accuracy = 0

    # model = net
    for texts, labels, lengths in test_loader:


        # print(f'texts.shape : {texts.shape}')
        # print(f'labels.shape : {labels.shape}')
        # print(f'length.shape : {lengths.shape}')
        # print(f'length : {lengths}')

        # Packed Sequence Processing
        lengths, sort_idx = torch.sort(lengths, descending= True)
        texts = texts[sort_idx]
        labels = labels[sort_idx]
        # print(f' After sorting length : {lengths}')
        # break

        output = model.forward(texts, h0 = None, l = lengths)
        loss = criterion(output, labels.type(torch.FloatTensor))
        test_loss += loss.item()

        ps = torch.sigmoid(output)
        ps_class = (ps > 0.5).float()

        equals = ps_class == labels.type(torch.FloatTensor).view(*ps_class.shape)

        test_accuracy += torch.mean(equals.type(torch.FloatTensor))


    test_loss = test_loss/len(test_loader)
    test_accuracy = test_accuracy / len(test_loader)

    return test_loss, test_accuracy

# Step5 Define Training

def training(model, criterion, optimizer, train_loader, test_loader, epochs = 3, print_every = 40):
    train_losses, test_losses = [], []
    test_loss_min = np.Inf
    print(f'epochs : {epochs}')
    print(f'train_loader : {len(train_loader)}')

    for epoch in range(epochs):
        steps = 0
        running_loss = 0
        model.train()

        for texts, labels, lengths in train_loader:
            steps += 1
            # Step5-1 Initialzing Optimizer
            optimizer.zero_grad()

            # Packed Sequence Processing
            lengths, sort_idx = torch.sort(lengths, descending=True)
            texts = texts[sort_idx]
            labels = labels[sort_idx]

            # Step5-2 Get prediction
            output = model.forward(texts, h0=None, l=lengths)

            # Step5-3 Calculate loss
            loss = criterion(output, labels.type(torch.FloatTensor))
            running_loss += loss.item()

            # Step5-1 loss backward
            loss.backward()

            # Step5-1 update optimizer
            optimizer.step()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_los, test_acc = validation(model, criterion, test_loader)

                print(f'Epochs {epoch + 1}/{epochs} \t'
                      f'Steps {steps} / {len(train_loader)} \t'
                      f'Training Loss {running_loss/steps:0.03f} \t'
                      f'Test Loss {test_los:0.03f} \t'
                      f'Test Accuracy {test_acc:0.03f}')

                model.train()
        else:
            model.eval()
            with torch.no_grad():
                test_los, test_acc = validation(model, criterion, test_loader)

            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_los)

            print(f'Epochs {epoch + 1}/{epochs} \t'
                  f'Training Loss {train_losses[-1]:0.03f} \t'
                  f'Test Loss {test_losses[-1]:0.03f} \t'
                  f'Test Accuracy {test_acc:0.03f}')

            if test_los <= test_loss_min:
                print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,test_los))
                torch.save(model.state_dict(), 'sent_imdb.pt')
                test_loss_min = test_los

            model.train()

    return model, train_losses, test_losses

model, train_losses, test_losses = training(net, criterion, optimizer, train_loader, test_loader, epochs = 5, print_every = 40)

# Step6 plot losses
plt.plot(train_losses, label = 'Training loss')
plt.plot(test_losses, label = 'Test_loss')
plt.legend(frameon=False)


#########################
##### Inference #########
#########################

def predict(model, review):

    vocab_path = pathlib.Path('/Users/philhoonoh/Desktop/scribbling/Udacity/Intro_to_Deep_Learning_with_PyTorch/3_recurrent-neural-networks/data/aclImdb/imdb.vocab')
    vocab_array = vocab_path.open(encoding='utf-8').read().strip().splitlines()
    vocab_dict = dict((w, i + 1) for (i,w) in enumerate(vocab_array))

    token_index = text2ids(review, vocab_dict)
    texts, lengths = list2tensor(token_index, max_len=100, padding=True)
    texts = texts.unsqueeze(dim = 0)
    lengths = torch.tensor(lengths).view(1)
    # texts.shape
    # lengths.shape


    model.eval()
    with torch.no_grad():

        # Packed Sequence Processing
        lengths, sort_idx = torch.sort(lengths, descending=True)
        texts = texts[sort_idx]

        output = model.forward(texts, h0=None, l=lengths)

        ps = torch.sigmoid(output).squeeze()

        print('Prediction value, pre-rounding: {:.6f}'.format(ps))

        if (ps.item() > 0.5):
            print("Positive review detected!")
        else:
            print("Negative review detected.")

review = """test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'"""
review_pos = """I Like this Love this movie, all time favorite"""

predict(model, review)
predict(model, review_pos)

# model.eval()
# with torch.no_grad():
#     test_loss, test_accuracy = validation(model, criterion, test_loader)


