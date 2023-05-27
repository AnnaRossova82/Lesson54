#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import scipy
 

import torch 
import torchvision 
import torch.nn as nn 
import torch.utils.data as data 
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable 

 

from torch.utils.tensorboard import SummaryWriter 
import datetime,os
import time
 
import seaborn as sns
sns.set_style('darkgrid')


# In[5]:


inputSize = 784
numClasses = 10
numEpochs = 10
learningRate = 0.001 
hidden = 500


# In[9]:


# Завантажимо MNIST
batchSize = 100

# Навчальна вибірка
trainDataset = dsets.MNIST(root='C:/Users/38098', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
# Тестова вибірка
testDataset = dsets.MNIST(root='C:/Users/38098', 
                           train=False, 
                           transform=transforms.ToTensor())

# Dataset Loader (підготовка даних для мережі)
trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, # Який датасет
                                           batch_size=batchSize, # На скільки batch поділено
                                           shuffle=True) 

testLoader = torch.utils.data.DataLoader(dataset=testDataset, #  Який датасет
                                          batch_size=batchSize, # На скільки batch поділено
                                          shuffle=False)


# In[10]:


# Подивимося на нашу вибірку
dataIter = iter(trainLoader) # Якою вибіркою пройдемося
trainX, trainY = next(dataIter) # Привласнюємо поточний batch


# In[31]:


trainX.view(-1, 28*28).shape # view операція ідентична reshape, за винятком, що вона змінює розмір масиву тільки для даної ітерації


# In[32]:


trainX.shape


# In[33]:


plt.figure(figsize=(14,7))
plt.imshow(trainX[1, 0], cmap='gray')
plt.show()


# In[63]:


# Тут пробувала додати третій шар як у завданні - ала дає помилку розмірності; і не ясно який датасет з лекції використовувати - 
# там був з трьома класами, чи цей, то я невдало спробувала обидва і просто прогнала весь код
class Classification(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__() 
        self.layer1 = nn.Linear(inputSize, hidden_size) # перший шар - лінійний
        self.relu = nn.ReLU() 
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1) # Оскільки завдання класифікації, то функція активації softmax

    def forward(self, x): # Тут ми прописуємо принципи, за якими дані проходитимуть через мережу
        out = self.layer1(x)  # вихід першого шару
        out = self.relu(out) # застосовуємо функцію активації до виходу першого шару
        out = self.layer2(out) # застосовуємо функцію активації до виходу першого шару
        out = self.softmax(out) # застосовуємо функцію активації до другого шару
        return out

model = Classification(inputSize, hidden, numClasses) # Створюємо об'єкт нашої повної мережі


# In[51]:


trainX.view(-1, 28*28).shape


# In[52]:


trainX.size()


# In[64]:


criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)  # оптимізатор
losses = [] 
model.train() 


# In[65]:


for epoch in range(numEpochs): # кількість епох
    lossTot = 0 # втрати в сукупності

    for i, (images, labels) in enumerate(trainLoader): # проходимося за всіма даними в batch
        images = images.view(-1, 28*28) # наводимо до правильного формату для сітки
        optimizer.zero_grad() # обнулюємо градієнт
        outputs = model(images) # тут наше передбачення
        loss = criterion(outputs.log(), labels) # рахуємо похибку 
        loss.backward()  # зворотне поширення. 
                         # x.grad += dloss/dx для всіх параметрів x

        lossTot +=loss.detach().data # інкремент помилки
        
        optimizer.step() # наступний крок спуску

    losses.append(lossTot/len(trainDataset)) # обчислюємо середню помилку та додаємо до списку
    print('Епоха: [%d/%d], Похибка: %.4f' 
           % (epoch+1, numEpochs, loss))
plt.figure(figsize=(14,7))
plt.plot(losses) # Графік нашого навчання
plt.show()


# In[66]:


plt.figure(figsize=(14,7))
plt.imshow(trainX[16, 0], cmap='gray')
plt.show()
print('Правильна відповідь', trainY[9])


# In[57]:


x = model(trainX[9].view(-1, 784)).detach()
print(x.data)
print(torch.max(x.data,dim=1)) # відповідь у вигляді ймовірностей


# In[58]:


correct = 0
total = 0

model.eval() # режим перевірки

for images, labels in testLoader: # ітеруємо по перевірочному датасету
    images = images.view(-1, 28*28) # наводимо до потрібного формату
    result = labels
    outputs = model(images) # робимо прогноз
    _, predicted = torch.max(outputs.data, 1) # _ максимальне значення пропускаємо, нас цікавить, що це за цифра
    total += labels.size(0) # 0 - перше/єдине значення
    correct += (predicted == labels).sum() 
    
print('Точність для 10000 картинок: %d %%' % (100 * correct // total))


# In[59]:


print(outputs[0], torch.max(outputs.data, 1))
print(result)


# In[60]:


print(model)


# In[61]:


import torchviz


# In[62]:


torchviz.make_dot(model(images), params=dict(model.named_parameters()))


# In[ ]:





# In[ ]:


# І CIFAR10 щоб подивитись 


# In[5]:


from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
batch_size = 10
number_of_labels = 10 

# Create an instance for training. 
# When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally. 
train_set =CIFAR10(root="./data",train=True,transform=transformations,download=True)

# Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_loader)*batch_size)

# Create an instance for testing, note that train is set to False.
# When we run this code for the first time, the CIFAR10 test dataset will be downloaded locally. 
test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)

# Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[7]:


trainDataset = dsets.CIFAR10(root= 'C:/Users/38098/Downloads', 
                              train=True, 
                              transform=transforms.ToTensor(),
                              download=True) 


# In[67]:


image, label = trainDataset[9]
print(image.size())
print(image[1].size())
print(type(image)) 
print(label)


# In[9]:


im = image.numpy()[1]
x = Image.fromarray((im * 255).astype(np.uint8))
plt.imshow(x.convert('RGBA'))
plt.show()


# In[ ]:




