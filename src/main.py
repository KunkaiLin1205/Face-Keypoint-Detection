import numpy as np
from matplotlib import pyplot as plt
import torch
import skimage.io as skio
from skimage.transform import resize
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

# load gray scale image, nose_keypoint and landmarks, modify the example code given
def load(i, j):
    # construct the path
    path = "./imm_face_db/" + "{:02d}-{:d}{}".format(i, j, "m")
    if (not os.path.exists(path + ".asf")):
        path = "./imm_face_db/" + "{:02d}-{:d}{}".format(i, j, "f")
    
    # load all facial keypoints/landmarks
    file = open(path + ".asf")
    points = file.readlines()[16:74]
    landmark = []

    for point in points:
        x, y = point.split('\t')[2:4]
        landmark.append([float(x), float(y)])

    # the nose keypoint
    nose_keypoint = np.array(landmark).astype('float32')[-6]

    # load gray scale image and resize it
    im_gray = skio.imread(path + ".jpg", as_gray=True)

    return im_gray, nose_keypoint, np.array(landmark).astype('float32')

#################### part 1 classes and helper functions #################### 
class NoseKeypointDataset(Dataset):

    def __init__(self, images, landmarks, transform=None):
        self.images = images
        self.landmarks = landmarks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        landmark = self.landmarks[idx]
        sample = {"image": image, "landmarks": landmark}

        if self.transform:
            sample = self.transform(sample)

        return sample

# normalize the pixel values of image
class Normalize(object):
    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        image_normalized = image.astype(np.float32) / 255 - 0.5  
        return {"image": image_normalized, "landmarks": landmarks}

# resize the image
class Resize(object):
    def __init__(self, h, w):
        self.height = h
        self.width = w
    
    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        image_resized = resize(image, (self.height, self.width))
        return {"image": image_resized, "landmarks": landmarks}

class Net_Nose1(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 10, 3)
    self.conv2 = nn.Conv2d(10, 20, 3)
    self.conv3 = nn.Conv2d(20, 26, 3)
    self.conv4 = nn.Conv2d(26, 32, 3)

    self.fc1 = nn.Linear(32 * 1 * 3, 48)
    self.fc2 = nn.Linear(48, 2)
    
  def forward(self, X):
    X = F.max_pool2d(F.relu(self.conv1(X)), 2)
    X = F.max_pool2d(F.relu(self.conv2(X)), 2)
    X = F.max_pool2d(F.relu(self.conv3(X)), 2)
    X = F.max_pool2d(F.relu(self.conv4(X)), 2)

    X = torch.flatten(X, 1)
    X = F.relu(self.fc1(X))
    X = self.fc2(X)

    return X

class Net_Nose2(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 12, 5)
    self.conv2 = nn.Conv2d(12, 24, 3)
    self.conv3 = nn.Conv2d(24, 32, 3)
    self.conv4 = nn.Conv2d(32, 16, 3)

    self.fc1 = nn.Linear(16 * 1 * 3, 20)
    self.fc2 = nn.Linear(20, 2)
    
  def forward(self, X):
    X = F.max_pool2d(F.relu(self.conv1(X)), 2)
    X = F.max_pool2d(F.relu(self.conv2(X)), 2)
    X = F.max_pool2d(F.relu(self.conv3(X)), 2)
    X = F.max_pool2d(F.relu(self.conv4(X)), 2)

    X = torch.flatten(X, 1)
    X = F.relu(self.fc1(X))
    X = self.fc2(X)

    return X

#################### part 2 classes and helper functions #################### 
class FaceKeypointDataset(Dataset):

    def __init__(self, images, landmarks, transform=None):
        self.images = images
        self.landmarks = landmarks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        landmark = self.landmarks[idx]
        sample = {"image": image, "landmarks": landmark}

        if self.transform:
            sample = self.transform(sample)

        return sample

# custome transform to randomly shift image and its landmarks in -10 to 10 pixel in both x and y direction
class RandomShift(object):
    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]
        im_shifted = np.zeros_like(image)
        x_shift = random.randint(-10, 10)
        y_shift = random.randint(-10, 10)
        im_shifted[max(0, y_shift):min(h, h+y_shift), 
                    max(0, x_shift):min(w, w+x_shift)] = image[max(0, -y_shift):min(h, h-y_shift),
                                                                    max(0, -x_shift):min(w, w-x_shift)]
        landmarks_shifted = landmarks
        landmarks_shifted[:, 0] += x_shift * np.ones_like(landmarks_shifted[:, 0]) / 240
        landmarks_shifted[:, 1] += y_shift * np.ones_like(landmarks_shifted[:, 1]) / 180

        return {"image": im_shifted, "landmarks": landmarks_shifted}

# custome transform to randomly rotate image and its landmarks in -15 to 15 degree angles 
class RandomRotate(object):
    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]
        angle = random.randint(-15, 15)
        keypoints = KeypointsOnImage([Keypoint(x=pt[0] * w, y=pt[1] * h) for pt in landmarks], shape = image.shape)
        seq = iaa.Sequential([
            iaa.Affine(
                rotate = angle
            )
        ])

        im_rotated, landmarks_rotated = seq(image=image, keypoints=keypoints)
        landmarks_rotated_scaled = np.array([[pt.x / w, pt.y / h] for pt in landmarks_rotated]).astype(np.float32)
        return {"image": im_rotated, "landmarks": landmarks_rotated_scaled}

class Net_Face(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 6, 7)
    self.conv2 = nn.Conv2d(6, 12, 5)
    self.conv3 = nn.Conv2d(12, 18, 3)
    self.conv4 = nn.Conv2d(18, 20, 3)
    self.conv5 = nn.Conv2d(20, 24, 3)
    self.conv6 = nn.Conv2d(24, 32, 3)
    self.fc1 = nn.Linear(32 * 19 * 26, 1976)
    self.fc2 = nn.Linear(1976, 58 * 2)
    
  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.relu(self.conv2(X))
    X = F.relu(self.conv3(X))
    X = F.max_pool2d(F.relu(self.conv4(X)), 2)
    X = F.max_pool2d(F.relu(self.conv5(X)), 2)
    X = F.max_pool2d(F.relu(self.conv6(X)), 2)

    X = torch.flatten(X, 1)
    X = F.relu(self.fc1(X))
    X = self.fc2(X)

    return X

if __name__ == "__main__":
    # to prevent crash when running locally
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    # set the seed for randomness
    torch.manual_seed(1205)
    random.seed(1205)

    # load the data needed for part 1 and part 2
    images_list = []
    landmarks_nose_list = []
    landmarks_list = []
    for i in range(40):
        for j in range(6):
            im, landmarks_nose, landmarks = load(i + 1, j + 1)
            images_list.append(im)
            landmarks_nose_list.append(landmarks_nose)
            landmarks_list.append(landmarks)

    #################### part 1 #################### 

    # create training and validation dataset and do the data augmentation
    ts = transforms.Compose([Resize(60, 80), Normalize()])
    training_dataset = NoseKeypointDataset(images_list[:192], landmarks_nose_list[:192], transform=ts)
    validation_dataset = NoseKeypointDataset(images_list[192:], landmarks_nose_list[192:], transform=ts)

    # create dataloaders, using batch size of 1
    training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle = True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle = True)

    # sampled the first 6 images with ground-truth nose keypoint in dataloader
    fig, ax = plt.subplots(1, 6, figsize=(24,18))
    for i, sample_batched in enumerate(training_dataloader):
        images, landmarks = sample_batched["image"], sample_batched["landmarks"]
        ax[i].imshow(images[0], cmap="gray")
        ax[i].set_title("Image{}".format(i + 1))
        ax[i].scatter(landmarks[0][0] * 80, landmarks[0][1] * 60, s=100, c="green")

        if i == 5:
            break
    
    plt.tight_layout()
    plt.savefig("./results/nose_ground-truth.jpg")
    plt.close()


    # set parameters
    epochs = 25
    # a list learning rate for varying hyperparameters
    lr_list = [0.001, 0.0001, 0.01]
    # 2 NN with different architectures
    net_list = [Net_Nose1, Net_Nose2]

    models = []
    for net in net_list:
        for lr in lr_list:
            model = net()

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            training_loss = []
            validation_loss = []

            for epoch in range(epochs):
                losses = []
                for sample_batched in training_dataloader:
                    optimizer.zero_grad()
                    x, y = sample_batched["image"], sample_batched["landmarks"]
                    x = x.unsqueeze(1)
                    pred = model(x)
                    loss = criterion(pred, y)
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()

                training_loss.append(np.mean(losses))
                #evaluate validation loss
                losses = []
                for sample_batched in validation_dataloader:
                    x, y = sample_batched["image"], sample_batched["landmarks"]
                    x = x.unsqueeze(1)
                    pred = model(x)
                    loss = criterion(pred, y)
                    losses.append(loss.item())
                validation_loss.append(np.mean(losses))

            models.append(model)
            # plot the training and validation loss
            plt.figure()
            plt.plot(range(epochs), training_loss, label="training loss")
            plt.plot(range(epochs), validation_loss, label="validation loss")
            plt.title("Training and Validation loss vs. epochs" + " (lr={}, {})".format(lr, net.__name__))
            plt.xlabel("Epochs")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig("./results/loss_{}_{}.jpg".format(lr, net.__name__))
            plt.close()
        
    # use the first network model (Net_Nose1 with lr = 0.001)
    model_chosen = models[0]

    # apply the model on the first 6 images in validation set
    for i, sample_batched in enumerate(validation_dataloader):
        x, y = sample_batched["image"], sample_batched["landmarks"]
        x = x.unsqueeze(1)
        pred = model_chosen(x)
        pred = np.array(pred.detach())
        plt.figure()
        plt.imshow(sample_batched["image"][0], cmap="gray")
        plt.scatter(y[0][0] * 80, y[0][1] * 60, s=100, c="green")
        plt.scatter(pred[0][0] * 80, pred[0][1] * 60, s=100, c="red")
        plt.tight_layout()
        plt.savefig("./results/nose_prediction{}.jpg".format(i + 1))
        plt.close()
        
        if i == 5:
            break

    #################### part 2 #################### 
    
    # create training and validation dataset and do the data augmentation
    ts = transforms.Compose([Resize(180, 240),
                            RandomRotate(),
                            RandomShift(),
                            Normalize()])
    training_dataset = FaceKeypointDataset(images_list[:192], landmarks_list[:192], transform=ts)
    validation_dataset = FaceKeypointDataset(images_list[192:], landmarks_list[192:], transform=ts)

    # create dataloaders, using batch size of 8
    BATCH_SIZE = 8
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle = True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle = True)


    # sampled the first 6 images with ground-truth face keypoints in dataloader
    fig, ax = plt.subplots(1, 6, figsize=(24,18))
    for i, sample_batched in enumerate(training_dataloader):
        images, landmarks = sample_batched["image"], sample_batched["landmarks"]

        for j in range(6):
            ax[j].imshow(images[j], cmap="gray")
            ax[j].set_title("Image{}".format(j + 1))
            ax[j].scatter(landmarks[j, :, 0] * 240, landmarks[j, :, 1] * 180, s=10, c="green")

        break
        
    plt.tight_layout()
    plt.savefig("./results/face_ground-truth.jpg")
    plt.close()

    # train the model
    model = Net_Face()

    lr = 0.00001
    epochs = 25

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_loss = []
    validation_loss = []

    for epoch in range(epochs):
        losses = []
        for sample_batched in training_dataloader:
            optimizer.zero_grad()
            x, y = sample_batched["image"], sample_batched["landmarks"]
            x = x.unsqueeze(1)
            pred = model(x)
            pred = torch.reshape(pred, (pred.shape[0], 58, 2))
            loss = criterion(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        training_loss.append(np.mean(losses))

        #evaluate validation loss
        losses = []
        for sample_batched in validation_dataloader:
            x, y = sample_batched["image"], sample_batched["landmarks"]
            x = x.unsqueeze(1)
            pred = model(x)
            pred = torch.reshape(pred, (pred.shape[0], 58, 2))
            loss = criterion(pred, y)
            losses.append(loss.item())
        validation_loss.append(np.mean(losses))

    # plot the training and validation loss
    plt.figure()
    plt.plot(range(epochs), training_loss, label="training loss")
    plt.plot(range(epochs), validation_loss, label="validation loss")
    plt.title("Training and Validation loss vs. epochs" + " (lr={}, {})".format(lr, Net_Face.__name__))
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("./results/loss_{}_{}.jpg".format(lr, Net_Face.__name__))
    plt.close()

    # apply the model on images in 2 batches in validation set
    cnt = 1
    for i, sample_batched in enumerate(validation_dataloader):
        if i == 1 or i == 2:
            x, y = sample_batched["image"], sample_batched["landmarks"]
            x = x.unsqueeze(1)
            pred = model(x)
            pred = torch.reshape(pred, (pred.shape[0], 58, 2))

            for j in range(BATCH_SIZE):
                plt.figure()
                plt.imshow(sample_batched["image"][j], cmap="gray")
                plt.scatter(y[j, :, 0] * 240, y[j, :, 1] * 180, s=50, c="green")
                plt.scatter(pred.detach()[j, :, 0] * 240, pred.detach()[j, :, 1] * 180, s=50, c="red")
                plt.savefig("./results/face_prediction{}.jpg".format(cnt))
                plt.close()
                cnt += 1

    
    # visualize the learned filters of the first two conv layers
    fig, ax = plt.subplots(1, 6, figsize=(20, 20))
    for i, filter in enumerate(model.conv1.weight.detach()):
        ax[i].imshow(filter.permute(1, 2, 0), cmap="gray")
        ax[i].set_title("{}".format(i + 1))
        ax[i].axis("off")
    plt.tight_layout()
    plt.savefig("results/learned_filters_conv1.jpg")
    plt.close()

    fig, ax = plt.subplots(12, 6, figsize=(20, 20))
    for i, filters in enumerate(model.conv2.weight):
        for j, filter in enumerate(filters): 
            ax[i][j].imshow(filter.detach(), cmap="gray")
            ax[i][j].set_title("{}-{}".format(i + 1, j + 1))
            ax[i][j].axis("off")
    plt.tight_layout()
    plt.savefig("results/learned_filters_conv2.jpg")
    plt.close()