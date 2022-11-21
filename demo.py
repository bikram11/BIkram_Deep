# %%


from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import glob
from Dual_model import config
from Dual_model.unet_model import UNet
from tqdm import tqdm
import torch
# %%
def create_image_mask_subarray(image_directory):
    image_names = glob.glob(image_directory+"*.png")
    image_names.sort()
   
    images = [cv2.imread(img,0) for img in image_names]

    image_dataset = np.array(images)
    # image_dataset = np.expand_dims(image_dataset,axis=3)
    return image_dataset


# %%
mask_directory = 'train_videos/000/maskedImages/'
image_directory = 'train_videos/000/rawImages/'

image_dataset=create_image_mask_subarray(image_directory)

mask_dataset=create_image_mask_subarray(mask_directory)

# %%
from PIL import Image
path = 'train_annotations'

dir_list = os.listdir(path)
for inddir in tqdm(dir_list):
        if(int(inddir[0:3])>0 and int(inddir[0:3])<2):
            mask_directory = 'train_videos/'+inddir[0:3]+"/maskedImages/"
            image_directory = 'train_videos/'+inddir[0:3]+"/rawImages/"

            image_dataset=np.concatenate((image_dataset, create_image_mask_subarray(image_directory)))
            
            mask_dataset=np.concatenate((mask_dataset, create_image_mask_subarray(mask_directory)))





# %%

# %%



# %%




# %%


from torchvision import transforms
        
#### Testing generator, generates augmented images
transforms = transforms.Compose([
	transforms.ToTensor()])


# %%
#Normalize images
image_dataset = transforms(image_dataset) /255.  #Can also normalize or scale using MinMax scaler
#Do not normalize masks, just rescale to 0 to 1.
mask_dataset = transforms(mask_dataset) /255.  #PIxel values will be 0 or 1

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 42)


# # %%
# import random

# image_number = random.randint(0, len(X_train)-1)
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(X_train[image_number,:,:,0], cmap='gray')
# plt.subplot(122)
# plt.imshow(y_train[image_number,:,:,0], cmap='gray')
# plt.show()


# %%
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
# IMG_CHANNELS = image_dataset.shape[3]

# input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)



# %%
unet = UNet().to(config.DEVICE)


# %%
from torch.utils.data import DataLoader
trainLoader = DataLoader((X_train,y_train), shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader((X_test,y_test), shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())

# %%
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(X_train) // config.BATCH_SIZE
testSteps = len(X_test) // config.BATCH_SIZE
print("testSteps"+str(testSteps))
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# %%
import time
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE,dtype=torch.float), y.to(config.DEVICE, dtype=torch.float))
		# perform a forward pass and calculate the training loss
        
		pred = unet(x)
		# print(pred)
		loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for (x1, y1) in testLoader:
			# send the input to the device
			(x1, y1) = (x1.to(config.DEVICE), y1.to(config.DEVICE))
			# make the predictions and calculate the validation loss
			pred = unet(x1)

			loss=lossFunc(pred,y1)

			totalTestLoss += loss
			
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	# update our training historyß
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))


# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
# serialize the model to disk
torch.save(unet, config.MODEL_PATH)

# # %%
# model.save('25epoch_lead_vehicle_segmentation.hdf5')

# # %%
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # %%
# from keras.models import load_model
# model = load_model('25epoch_lead_vehicle_segmentation.hdf5', compile=False)


# # %%

# y_pred=model.predict(X_test)
# y_pred_thresholded = y_pred > 0.5

# # %%
# from tensorflow.keras.metrics import MeanIoU

# # %%
# n_classes = 2
# IOU_keras = MeanIoU(num_classes=n_classes)  
# IOU_keras.update_state(y_pred_thresholded, y_test)
# print("Mean IoU =", IOU_keras.result().numpy())

# # %%
# threshold = 0.5
# test_img_number = random.randint(0, len(X_test)-1)
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# test_img_input=np.expand_dims(test_img, 0)
# print(test_img_input.shape)
# prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
# print(prediction.shape)

# plt.figure(figsize=(16, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='gray')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(prediction, cmap='gray')

# plt.show()

# %%



