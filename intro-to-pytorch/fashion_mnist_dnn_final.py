import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

###################
### Data Loading ##
###################

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle = True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/FF_MNIST_data/', download=True, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

###################
#### Check Data ###
###################

images, labels = next(iter(trainloader))
print(images.shape)
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

#########################
#### Model with Class ###
#########################

from torch import nn
from torch import optim
import torch.nn.functional as F

from collections import OrderedDict

# Step1 Build the Model

# Using nn.Module (accepts nn.functional & nn.Module)
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)

##############################
#### Model with Sequential ###
##############################

# Using Sequential Module (accepts only nn.Module)
# classifier2 = nn.Sequential(OrderedDict([
#     ('fc1',nn.Linear(input_dim, hidden_layers[0])),
#     ('act1',nn.ReLU()),
#     ('dropout1', nn.Dropout(p = 0.2)),
#     ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
#     ('act2',nn.ReLU()),
#     ('dropout2', nn.Dropout(p = 0.2)),
#     ('fc3', nn.Linear(hidden_layers[1], hidden_layers[2])),
#     ('act3',nn.ReLU()),
#     ('dropout3', nn.Dropout(p = 0.2)),
#     ('fc4', nn.Linear(hidden_layers[2], output_dim)),
#     ('act4',nn.ReLU()),
#     ('softmax', nn.LogSoftmax(dim = 1))
# ]))

input_dim = 784
hidden_layers = [256, 128, 64]
output_dim = 10
network = Network(input_dim, output_dim, hidden_layers)

# Step2 Set loss
criterion = nn.NLLLoss()

# Step3 Set Optimizer
optimizer = optim.SGD(network.parameters(), lr = 0.003)

# Step4 Define Validation
def validation(model, testloader, criterion):
    test_loss = 0
    test_accuracy = 0

    for images, labels in testloader:
        images = images.view(images.shape[0], -1)

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        test_loss += loss.item()

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor))

    return test_loss, test_accuracy


# Step5 Define Train
def train(model, trainloader, testloader, criterion, optimizer, epochs = 10, print_every=40):
    test_losses, train_losses = [], []

    for epoch in range(epochs):
        model.train()
        steps = 0
        running_loss = 0

        for images, labels in trainloader:
            steps +=1

            images = images.view(images.shape[0], -1)

            #Step5-1 Initialzing Optim
            optimizer.zero_grad()

            #Step5-2 Prediction
            log_ps = model(images)

            #Step5-3 Calculate Loss
            loss = criterion(log_ps, labels)
            running_loss += loss.item()

            #Step5-4 backward loss
            loss.backward()

            #Step5-5 update Gradient
            optimizer.step()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, test_accuracy = validation(model, testloader, criterion)

                print(f'Epochs {epoch+1}/{epochs} \t'
                      f'Step {steps}/{len(trainloader)} \t'
                      f'Training Loss {running_loss/steps:.3f} \t'
                      f'Test Loss {test_loss/len(testloader):.3f} \t'
                      f'Test Accuracy {test_accuracy/len(testloader):.3f}')

                model.train()
        else:
            model.eval()
            with torch.no_grad():
                test_loss, test_accuracy = validation(model, testloader, criterion)

            test_losses.append(test_loss / len(testloader))
            train_losses.append(running_loss / len(trainloader))

            print(f'Epochs {epoch + 1}/{epochs} \t'
                  f'Training Loss {train_losses[-1]:.3f} \t'
                  f'Test Loss {test_losses[-1]:.3f}\t'
                  f'Test Accuracy {test_accuracy/len(testloader):.3f}')

            model.train()

    return train_losses, test_losses

train_losses, test_losses = train(network, trainloader, testloader, criterion, optimizer, epochs = 10, print_every=40)


# Step6 plot losses
plt.plot(train_losses, label = 'Training loss')
plt.plot(test_losses, label = 'Test_loss')
plt.legend(frameon=False)

# Step7 Inference
images, label = next(iter(testloader))

image = images[0]

network.eval()
with torch.no_grad():
    image = image.view(image.shape[0], -1)
    log_ps = network(image)

ps = torch.exp(log_ps)

import numpy as np
def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

view_classify(image.resize_(1, 28, 28), ps, version='Fashion')

################################
#### Saving & Loading Models ###
################################

# Method1

# Saving and  Loading state_dict
torch.save(network.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth')

# Creating Same Architecture Network and putting state_dict on the Network
input_dim = 784
hidden_layers = [256, 128, 64]
output_dim = 10
classifier2 = Network(input_dim, output_dim, hidden_layers)
classifier2.load_state_dict(state_dict)

# Method2

checkpoint = {'input_dim': 784,
              'output_dim': 10,
              'hidden_layers': [each.out_features for each in network.hidden_layers],
              'state_dict': network.state_dict()}
torch.save(checkpoint, 'checkpoint2.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Network(checkpoint['input_dim'],
                             checkpoint['output_dim'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    return model

model = load_checkpoint('checkpoint2.pth')
print(model)