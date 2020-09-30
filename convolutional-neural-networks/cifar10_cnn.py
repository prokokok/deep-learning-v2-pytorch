import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

###################
##### Set GPU #####
###################

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    device = torch.device("cuda")
    print('CUDA is available!  Training on GPU ...')
else:
    device = torch.device("cpu")
    print('CUDA is not available.  Training on CPU ...')

###################
### Data Loading ##
###################

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# Adding Data Augmentation
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(30),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_data = datasets.CIFAR10('data', download = True, train = True, transform=transform)
test_data = datasets.CIFAR10('data', download=True, train = False, transform=transform)

# number of subprocesses to use for data loading
# how many samples per batch to load
# percentage of training set to use as validation
NUM_WORKERS = 0
BATCH_SIZE = 32
VALIDATION_SIZE = 0.2

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(VALIDATION_SIZE * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, sampler=valid_sampler, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, num_workers=NUM_WORKERS)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

#####################
#### Check Data 1 ###
#####################

def imshow(img):
    # un-normalize
    img = img / 2 + 0.5
    # convert from Tensor image
    plt.imshow(np.transpose(img, (1,2,0)))

images, labels = next(iter(train_loader))
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(BATCH_SIZE):
    ax = fig.add_subplot(2, BATCH_SIZE/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

#####################
#### Check Data 2 ###
#####################
rgb_img = np.squeeze(images[3])
channels = ['red channel', 'green channel', 'blue channel']

fig = plt.figure(figsize = (36, 36))
for idx in np.arange(rgb_img.shape[0]):
    ax = fig.add_subplot(1, 3, idx + 1)
    img = rgb_img[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(channels[idx])
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center', size=4,
                    color='white' if img[x][y]<thresh else 'black')

#########################
#### Model with Class ###
#########################

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


# Step1 Build Model
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)

        # channels_in, channels_out, kernel_size, padding(0:default), strdies(1:default)
        # output dim = (channels_out, height, width)
            # channels_out
            # height = (height of input - kernel + 2*padding)/strides + 1
            # weight = (height of input - kernel + 2*padding)/strides + 1

        self.conv1 = nn.Conv2d(3, 16, 3,  padding = 1)
        # (16, (32 - 3 + 2*1)/1 + 1 = 32, 32)
        self.bn1 = nn.BatchNorm2d(16)
        # (16, 32, 32)

        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        # (32, (32 - 3 + 2*1)/1 + 1 = 32, 32)
        self.bn2 = nn.BatchNorm2d(32)
        # (32, 32, 32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        # (64, 32, 32)
        self.bn3 = nn.BatchNorm2d(64)
        # (64, 32, 32)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # (64, 16, 16)

        self.fc1 = nn.Linear(64 * 16 * 16, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 64*16*16)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim = 1)

        return x

net = Network()

print('Creating Model')
net.to(device)

if train_on_gpu:
    print('Moving Model Tensors To GPU ...')
else:
    print('Model Tensors Staying On CPU ...')

# Step2. Set up Loss
criterion = nn.NLLLoss()

# Step3. Set up optim
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# Step4. Define Validation
def validation(model, valid_loader, criterion, train_on_gpu):
    val_loss = 0
    val_accuracy = 0

    for images, labels in valid_loader:
        if train_on_gpu:
            inputs, labels = images.to(device), labels.to(device)

        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        val_loss += loss.item()

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        val_accuracy += torch.mean(equals.type(torch.FloatTensor))

    val_loss = val_loss/len(valid_loader)
    val_accuracy = val_accuracy / len(valid_loader)

    return val_loss, val_accuracy

# Step5. Define Train
def training(model, train_loader, valid_loader, criterion, optimizer, epochs = 3, print_every = 40, train_on_gpu = False, device = torch.device("cpu")):
    train_losses, val_losses = [], []
    valid_loss_min = np.Inf

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        steps = 0

        for images, labels in train_loader:
            steps += 1

            if train_on_gpu:
                images, labels = images.to(device), labels.to(device)

            # Step5-1 Initializing Op
            optimizer.zero_grad()

            # Step5-2 Predict
            log_ps = model.forward(images)

            # Step5-3 Calculate Loss
            loss = criterion(log_ps, labels)
            running_loss += loss.item()

            # Step5-4 backward loss
            loss.backward()

            # Step5-5 update gradient
            optimizer.step()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    val_loss, val_accuracy = validation(model, valid_loader, criterion, train_on_gpu)

                print(f'Epochs {epoch + 1}/{epochs} \t'
                      f'Steps {steps}/{len(train_loader)}\t'
                      f'Training_Loss {running_loss/steps : 0.03f}\t'
                      f'Validation_Loss {val_loss:0.03f}\t'
                      f'Validation_Accuracy {val_accuracy:0.03f}')
                model.train()
        else:
            model.eval()
            with torch.no_grad():
                val_loss, val_accuracy = validation(model, valid_loader, criterion, train_on_gpu)

            train_losses.append(running_loss/len(train_loader))
            val_losses.append(val_loss)

            print(f'Epochs {epoch + 1}/{epochs} \t'
                  f'Training_Loss {train_losses[-1] : 0.03f}\t'
                  f'Validation_Loss {val_losses[-1]:0.03f}\t'
                  f'Validation_Accuracy {val_accuracy:0.03f}')

            if val_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_loss))
                torch.save(model.state_dict(), 'model_augmented.pt')
                valid_loss_min = val_loss

            model.train()

    return train_losses, val_losses


train_losses, val_losses = training(net, train_loader, valid_loader, criterion, optimizer, epochs = 3, print_every = 40, train_on_gpu = False, device = torch.device("cpu"))

# Step6 plot losses
plt.plot(train_losses, label = 'Training loss')
plt.plot(val_losses, label = 'Test_loss')
plt.legend(frameon=False)


################################
#### Saving & Loading Models ###
################################

model = Network()
model.load_state_dict(torch.load('model_augmented.pt'))

################################
#### Testing Trained Network ###
################################

def test(model, test_loader, criterion, train_on_gpu):

    if train_on_gpu:
        print('Loading Model On GPU')
        device = torch.device("cuda")
        model.to(device)
    else:
        print('Loading Model On CPU')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_accuracy = 0

        for images, labels in test_loader:
            if train_on_gpu:
                inputs, labels = images.to(device), labels.to(device)

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            test_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))

            if train_on_gpu:
                correct = np.squeeze(equals.numpy())
            else:
                correct = np.squeeze(equals.cpu().numpy())

            # calculate test accuracy for each object class
            for i in range(labels.shape[0]):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    test_loss = test_loss / len(test_loader)
    test_accuracy = test_accuracy / len(test_loader)

    print(f'Test_Loss : {test_loss : 0.03f}\t'
          f'Test_Accuracy : {test_accuracy:0.03f}')

    for i in range(10):
        if class_total[i] > 0:
            print(f'Test Accuracy of {classes[i]} : {class_correct[i]/class_total[i]:0.03f} ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'Test Accuracy of {classes[i]} : N/A (no training examples)')


test(model, test_loader, criterion, train_on_gpu = False)

################################
####### Inference Network ######
################################

# obtain one batch of test images
images, labels = next(iter(test_loader))

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
model.eval()
with torch.no_grad():
    log_ps = model.forward(images)
ps = torch.exp(log_ps)
top_p, top_class = ps.topk(1, dim = 1)

if train_on_gpu:
    preds = np.squeeze(top_class.numpy())
else:
    preds = np.squeeze(top_class.cpu().numpy())


# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(BATCH_SIZE):
    ax = fig.add_subplot(4, BATCH_SIZE/4, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))




