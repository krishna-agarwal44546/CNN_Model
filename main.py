import torch
from torch import nn
import pandas as pd
import numpy as np
import cv2
import datetime

# making a ResNET architecture based CNN with 2 residual blocks where each block has 2 convolutional layers and a skip connection. 
# The output of the last block is flattened and passed through a fully connected layer to get the final output.

class ResNET(nn.Module):
	def __init__(self):
		super().__init__()

		# Basic Block1 
		self.conv_1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1)
		self.conv_2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1)
		self.bn1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=1,stride=4,padding=0)

		#Basic Block 2
		self.conv_3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
		self.conv_4=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1)
		self.bn2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=4,padding=0)

		# Fully connected layer
		self.fc=nn.Linear(128*4*4,5)

	def block1(self,x:torch.Tensor)->torch.Tensor:
		x=x.permute(0,3,1,2)
		identity=x
		x=self.conv_1(x)
		x=nn.functional.relu(x)
		x=self.conv_2(x)
		identity=self.bn1(identity)
		x=x+identity
		x=nn.functional.relu(x)

		return x

	def block2(self,x:torch.Tensor)->torch.Tensor:
		identity=x
		x=self.conv_3(x)
		x=nn.functional.relu(x)
		x=self.conv_4(x)
		identity=self.bn2(identity)
		x=x+identity
		x=nn.functional.relu(x)
		return x
		
	def forward(self,x:torch.Tensor)->torch.Tensor:
		x=self.block1(x)
		x=self.block2(x)
		x=x.flatten(start_dim=1)
		x=self.fc(x)
		return x

model=ResNET()
print(list(model.parameters()))
print(model.state_dict())

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)


df=pd.read_csv('training_data.csv')
vals=df['patient_DR_Level'].values

y_train=[]
for i in range(1000):
	y_train.append(vals[i])

y_train=torch.tensor(y_train, dtype=torch.long)

#training loop
x_train=[]
for i in range(1000):
	path=df['image_path'][i]
	path=path.replace('\\','/')
	path=path.lstrip('/')
	path='regular_fundus_images/' + path.replace('regular-fundus-training','regular-fundus-training/Images')
	img=cv2.imread(path)
	img=cv2.resize(img,(64,64))
	img=torch.tensor(img,dtype=torch.float32)/255.0
	x_train.append(img)

x_train=torch.stack(x_train)
x_train=x_train.unsqueeze(1)

x_test=[]
for i in range(len(y_train),len(vals)):
	path=df['image_path'][i]
	path=path.replace('\\','/')
	path=path.lstrip('/')
	path='regular_fundus_images/' + path.replace('regular-fundus-training','regular-fundus-training/Images')
	img=cv2.imread(path)
	img=cv2.resize(img,(64,64))
	img=torch.tensor(img,dtype=torch.float32)/255.0
	x_test.append(img)
x_test=torch.stack(x_test)
x_test=x_test.unsqueeze(1)

y_test=[]
for i in range(len(y_train),len(vals)):
	y_test.append(vals[i])
y_test=torch.tensor(y_test, dtype=torch.long)

classes=['No DR','Mild','Moderate','Severe','Proliferative DR']
epochs=100
x=datetime.datetime.now()
for epoch in range(epochs):
	model.train()
	y_pred=model(x_train)
	loss=loss_fn(y_pred,y_train)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
y=datetime.datetime.now()
print(f"Training time: {y-x}")
model.eval()
with torch.no_grad():
	test_pred=model(x_test)
	probabilities=torch.softmax(test_pred,dim=1)
	val_loss=loss_fn(test_pred,y_test)

with torch.no_grad():
	train_pred=model(x_train)
	train_probabilities=torch.softmax(train_pred,dim=1)

train_correct=0
for i,probablity in enumerate(train_probabilities):
	if torch.argmax(probablity)==y_train[i].item():
		train_correct+=1
print(f"Train Accuracy : {train_correct/len(y_train)*100}%")



print(f"probablity: {probabilities}")
correct=0
for i,probablity in enumerate(probabilities):
	print(f"Predicted class: {classes[torch.argmax(probablity)]}, Probability: {torch.max(probablity).item()}  , Actual class : {classes[y_test[i].item()]}")
	if torch.argmax(probablity)==y_test[i].item():
		correct+=1

print(f"Accuracy : {correct/len(y_test)*100}%")