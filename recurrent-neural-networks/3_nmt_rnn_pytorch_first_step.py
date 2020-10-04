import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tqdm

###############################
##### Setting functions
###############################

import re
import collections
import itertools

remove_marks_regex = re.compile(
    "[\,\(\)\[\]\*:;¿¡]|<.*?>")
shift_marks_regex = re.compile("([?!\.])")

unk = 0
sos = 1
eos = 2

def normalize(text):
    text = text.lower()
    text = remove_marks_regex.sub("", text)
    text = shift_marks_regex.sub(" \1", text)

    return text

def parse_line(line):
    line = normalize(line.strip())
    src, trg = line.split('\t')[:2]
    # print(f'src : {src}')
    # print(f'trg : {trg}')
    src_tokens = src.strip().split()
    trg_tokens = trg.strip().split()
    return src_tokens, trg_tokens

def build_vocab(tokens):
    counts = collections.Counter(tokens)

    sorted_counts = sorted(counts.items(), key = lambda c : c[1], reverse = True)

    word_list = ["<UNK>", "<SOS>", "<EOS>"] + [x[0] for x in sorted_counts]

    word_dict = dict((w, i) for i, w in enumerate(word_list))

    return word_list, word_dict

def word2tensor(words, word_dict, max_len, padding = 0):

    words = words + ["<EOS>"]

    words = [word_dict.get(w,0) for w in words]
    seq_len = len(words)

    if seq_len < max_len + 1:
        words = words + [padding] * (max_len + 1 -seq_len)

    return torch.tensor(words, dtype=torch.int64), seq_len


# Testing Functions

# path = '/Users/philhoonoh/Desktop/scribbling/Udacity/Intro_to_Deep_Learning_with_PyTorch/3_recurrent-neural-networks/data/spa-eng copy/spa.txt'
# max_len = 100

# def filter_pair(p):
#     return not (len(p[0]) > max_len or len(p[1]) > max_len)
#
# with open(path, encoding='utf-8') as fp:
#     pairs = map(parse_line, fp)
#     pairs_2 = filter(filter_pair, pairs)
#     pairs_3 = list(pairs_2)
#
# src = [p[0] for p in pairs_3]
# trg = [p[0] for p in pairs_3]

###############################
##### Preparing Dataset #######
###############################

class TranslationPairDataset(Dataset):
    def __init__(self, path, max_len = 15):
        super(TranslationPairDataset, self).__init__()

        # 단어 수가 많으면 걸러냄
        def filter_pair(p):
            return not (len(p[0]) > max_len or len(p[1]) > max_len)

        with open(path, encoding='utf-8') as fp:
            pairs = map(parse_line, fp)
            pairs = filter(filter_pair, pairs)
            pairs = list(pairs)

        src = [p[0] for p in pairs]
        trg = [p[1] for p in pairs]

        self.src_word_list, self.src_word_dict = build_vocab(itertools.chain.from_iterable(src))
        self.trg_word_list, self.trg_word_dict = build_vocab(itertools.chain.from_iterable(trg))

        self.src_data = [word2tensor(words, self.src_word_dict, max_len, padding = 0) for words in src]
        self.trg_data = [word2tensor(words, self.trg_word_dict, max_len, padding = -100) for words in trg]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src, lsrc = self.src_data[idx]
        trg, ltrg = self.trg_data[idx]

        return src, lsrc, trg, ltrg

batch_size = 32
max_len = 10
path = '/Users/philhoonoh/Desktop/scribbling/Udacity/Intro_to_Deep_Learning_with_PyTorch/3_recurrent-neural-networks/data/spa-eng copy/spa.txt'
ds = TranslationPairDataset(path, max_len = max_len)
loader = DataLoader(ds, batch_size = batch_size, shuffle = True, num_workers = 2)

#########################
##### Build Model #######
#########################

# STEP1
# Encoder
class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim = 50, hidden_size = 50, num_layers = 1, dropout = 0.2):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first = True, dropout = dropout)

    def forward(self, x, h0 = None, l = None):
        x = self.emb(x)

        if l is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, l, batch_first = True)

        _, h = self.lstm(x, h0)

        return h

# Decoder
class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim = 50, hidden_size = 50, num_layers = 1, dropout = 0.2):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first = True, dropout = dropout)
        self.linear = nn.Linear(hidden_size, num_embeddings)

    def forward(self, x, h0 = None, l = None):
        x = self.emb(x)

        if l is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, l, batch_first = True)

        x, h = self.lstm(x, h0)

        if l is not None:
            x = nn.utils.rnn.pad_packed_sequence(x, batch_first = True, padding_value = 0)[0]

        x = self.linear(x)
        return x, h

enc = Encoder(len(ds.src_word_list), 50, 50, 1)
dec = Decoder(len(ds.trg_word_list), 50, 50, 1)

# Step2. Set up Optimizer
optimizer_enc = optim.Adam(enc.parameters(), lr = 0.002)
optimizer_dec = optim.Adam(dec.parameters(), lr = 0.002)

# Step3. Set up Loss
criterion = nn.CrossEntropyLoss()
def to2D(x):
    shapes = x.shape
    x = x.reshape(shapes[0] * shapes[1], -1)
    return x

# Step4. Define Validation (translate)
def validation(input_str, enc, dec, max_len=15):

    words = normalize(input_str).split()
    input_tensor, seq_len = word2tensor(words, ds.src_word_dict, max_len=max_len)
    input_tensor = input_tensor.unsqueeze(0)

    seq_len = [seq_len]

    sos_inputs = torch.tensor(sos, dtype=torch.int64)

    ctx = enc(input_tensor, l = seq_len)

    z = sos_inputs
    h = ctx
    result = []

    for i in range(max_len):
        o, h = dec(z.view(1,1), h)
        # get index
        wi = o.detach().view(-1).max(dim = 0)[1]

        if wi.item() == eos:
            break
        result.append(wi.item())

        z = wi

    return " ".join(ds.trg_word_list[i] for i in result)


# Step5. Define Training
def training(enc, dec, loader, optimizer_enc, optimizer_dec, criterion,epochs = 10, print_every = 32, clip = 5):
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0
        steps = 0

        for x, lx, y, ly in tqdm.tqdm(loader):
            steps +=1
            # Step1 Initailizing opt
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            # Step2 Prediction
            # Packed Sequence Processing for Encoder
            lx, sort_idx = torch.sort(lx, descending=True)
            x = x[sort_idx]
            y = y[sort_idx]
            ly = ly[sort_idx]

            # print('\nBefore Encoder')
            # print(f'x.shape : {x.shape}')
            # x.shape: torch.Size([32, 11])
            # print(f'lx.shape : {lx.shape}')
            # lx.shape: torch.Size([32])
            # print(f'y.shape : {x.shape}')
            # y.shape: torch.Size([32, 11])
            # print(f'ly.shape : {ly.shape}')
            # ly.shape: torch.Size([32])

            ctx = enc(x, l = lx)

            # print('\nAfter Encoder')
            # print(f'ctx.shape : {ctx[0].shape}')
            # ctx.shape: torch.Size([1, 32, 50])
            # print(f'ctx.shape : {ctx[1].shape}')
            # ctx.shape: torch.Size([1, 32, 50])

            # Packed Sequence Processing for Decoder
            ly, sort_idx = torch.sort(ly, descending=True)
            y = y[sort_idx]
            h0 = (ctx[0][:, sort_idx,:], ctx[1][:,sort_idx,:])

            # back propagation 에서 제거 & input_y 생성 : 맨 마지막 값 제거 (11 : max_len + <EOS> -> 10)
            # [Hello, I, know, <EOS>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>]
            # -> input_y = [Hello, I, know, <EOS>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>]
            input_y = y[:,:-1].detach()

            # padding 바꾸기
            input_y[input_y==-100] = 0

            # print('\nBefore Decoder')
            # print(f'y.shape : {y.shape}')
            # y.shape: torch.Size([32, 11])
            # print(f'input_y.shape : {input_y.shape}')
            # torch.Size([32, 10])

            # input_y 생성 에서 맨 마지막 값을 지웠으므로, l = ly - 1
            o, _  = dec(input_y, h0, l = ly-1)

            # print('\nAfter Encoder')
            # o.shape (batch_size, max_len in ly, num_embeddings)
            # print(f'o.shape : {o.shape}')
            # o.shape: torch.Size([32, 13, 13744])


            # Step3 Calculate Loss
            # back propagation 에서 제거 & target_y 생성 : 맨 앞 제거 (11 : max_len + <EOS> -> 10)
            # + o.shape 과 같게 하기
            # [Hello, I, know, <EOS>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>]
            # -> output_y = [I, know, <EOS>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>, <UNK>]
            output_y = y[:,1:max(ly)]
            # print('\nCalculate Loss')

            # o.shape (batch_size, max_len in ly, num_embeddings)
            # print(f'o.shape : {o.shape}')
            # o.shape: torch.Size([32, 10, 12303])
            # print(f'output_y.shape : {output_y.shape}')
            # output_y.shape: torch.Size([32, 10])
            # print(f'to2D(o[:]).shape : {to2D(o[:]).shape}')
            # to2D(o[:]).shape: torch.Size([320, 12303])
            # print(f'to2D(output_y).shape : {to2D(output_y).shape}')
            # to2D(output_y).shape: torch.Size([320, 1])
            loss = criterion(to2D(o[:]), to2D(output_y).squeeze())
            running_loss += loss.item()

            # Step4 loss backward grad_clip
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), clip)
            nn.utils.clip_grad_norm_(dec.parameters(), clip)

            # Step5 opt step
            optimizer_enc.step()
            optimizer_dec.step()

            if steps % print_every == 0:
                enc.eval()
                dec.eval()

                print(f'Epoch {epoch + 1}/{epochs} \t'
                      f'Steps {steps}/{len(loader)} \t'
                      f'Training Loss {running_loss / steps:0.03f}')

                input_str_lst = ["I am a student", "He likes to eat pizza.", "She is my mother"]
                with torch.no_grad():
                    for input_str in input_str_lst:
                        result = validation(input_str, enc, dec, max_len=15)
                        print(result)
                enc.train()
                dec.train()

        else:
            enc.eval()
            dec.eval()

            train_losses.append(running_loss / len(loader))

            print(f'Epoch {epoch + 1}/{epochs} \t'
                  f'Training Loss {train_losses[-1]:0.03f} \n')

            input_str_lst = ["I am a student", "He likes to eat pizza.", "She is my mother"]
            with torch.no_grad():
                for input_str in input_str_lst:
                    result = validation(input_str, max_len=15, enc=enc, dec=dec)
                    print(result)
            enc.train()
            dec.train()

training(enc, dec, loader, optimizer_enc, optimizer_dec, criterion,epochs = 10, print_every = 32, clip = 5)


