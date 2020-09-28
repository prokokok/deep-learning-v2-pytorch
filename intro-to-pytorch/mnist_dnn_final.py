import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

###################
### Data Loading ##
###################

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, ), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

###################
#### Check Data ###
###################
images, labels = next(iter(trainloader))
image = images[0]
plt.imshow(image.numpy().squeeze(), cmap='Greys_r')


##############################
#### Model with Sequential ###
##############################

input_dim = 784
hidden_sizes = [128, 64]
output_dim = 10

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_dim, hidden_sizes[0])),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout()),
                            ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout()),
                            ('output', nn.Linear(hidden_sizes[1], output_dim)),
                            ('logsoftmax', nn.LogSoftmax(dim = 1))]))

#########################
#### Model with Class ###
#########################

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, drop_p = 0.2):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_dim, hidden_sizes[0])])
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])

        self.hidden_layers.extend([nn.Linear(h1, h2)] for h1, h2 in layer_sizes)

        self.output = nn.Linear(hidden_sizes[-1], output_dim)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim = 1)

# Step2 Set loss
criterian = nn.NLLLoss()

# Step3 Set optimizer
optimzer = optim.SGD(classifier.parameters(), lr = 0.003)

# Step4 Define Validation

def validation(model, testloader, criterion):
    test_loss = 0
    test_accuracy = 0

    for images, labels in testloader:
        images = images.view(images.shape[0], -1)

        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        test_loss += loss.item()

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor))

    test_loss = test_loss/len(testloader)
    test_accuracy = test_accuracy/len(testloader)

    return test_loss, test_accuracy


# Step5 Define Train
def train(model, trainloader, testloader, optimizer, criterion, epochs = 10, print_every = 40):
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        running_loss = 0
        steps = 0

        for images, labels in trainloader:
            steps+=1
            images = images.view(images.shape[0], -1)

            # Step5-1 Initializing Optim
            optimizer.zero_grad()

            # Step5-2 Predict
            log_ps = model.forward(images)

            # Step5-3 Calculate Loss
            loss = criterion(log_ps, labels)
            running_loss+=loss

            # Step5-4 loss backward
            loss.backward()

            # Step5-5 Gradient Update
            optimizer.step()

            if steps % print_every == 0:

                model.eval()
                with torch.no_grad():
                    test_loss, test_accuracy = validation(model, testloader, criterion)

                print(f'Epochs {epoch+1}/{epochs} \t'
                      f'Steps {steps}/{len(trainloader)} \t'
                      f'Training Loss {running_loss/steps:0.3f} \t'
                      f'Test Loss {test_loss} \t'
                      f'Test Accuracy {test_accuracy}')

                model.train()
        else:
            model.eval()
            with torch.no_grad():
                test_loss, test_accuracy = validation(model, testloader, criterion)

            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss)

            print(f'Epochs {epoch + 1}/{epochs} \t'
                  f'Training Loss {train_losses[-1]:0.3f} \t'
                  f'Test Loss {test_losses[-1]} \t'
                  f'Test Accuracy {test_accuracy}')

            model.train()

    return train_losses, test_losses

train_losses, test_losses = train(classifier, trainloader, testloader, optimzer, criterian)

# Step6 Plot Losses
plt.plot(train_losses, label = 'Training loss')
plt.plot(test_losses, label = 'Test loss')
plt.legend(frameon = False)


# Step7 Inference
images, labels = next(iter(trainloader))
image = images[0].view(images[0].shape[0], -1)
print(image.shape)

classifier.eval()
with torch.no_grad():
    log_ps = classifier.forward(image)
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

view_classify(image.resize_(1, 28, 28), ps, version='MNIST')