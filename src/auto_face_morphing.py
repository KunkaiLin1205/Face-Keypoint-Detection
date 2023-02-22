import numpy as np
from matplotlib import pyplot as plt
import torch
import skimage.io as skio
from skimage.transform import resize
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from scipy.spatial import Delaunay
from skimage.draw import polygon
import imageio

#################### Helper functions in face morphing project #################### 

# get mean of the two points sets
def mean_points(lst1, lst2):
    return 0.5 * lst1 + 0.5 * lst2
    
# compute the affine transform matrix of t1 and t2 using homogeneous coordinates (with 3-vecotr for each point)
def computeAffine(t1, t2):
    ones = np.array([1, 1, 1])
    #represent input coordinates with 3-vectors
    t1_homo = np.vstack([t1.T, ones])
    t2_homo = np.vstack([t2.T, ones])
    T = np.matmul(t2_homo, np.linalg.inv(t1_homo))
    return T

# inverse warp the input image based on triangulation and transform matrices given
def inverse_warp(im, triangles, matrices):
    # make a copy of input triangles to prevent mutation
    triangles_copy = triangles.copy()
    
    res = np.ones_like(im)
    # traverse each triangle to warp the image
    for t, T in zip(triangles_copy, matrices):
        # generate the mask 
        rr, cc = polygon(t[:, 1], t[:, 0])
        mask = np.zeros_like(im[:, :, 0])
        mask[rr, cc] = 1

        # get the coordinates of points on the warped image and append ones to make each vector a 3-vector
        pts_warped = np.where(mask)
        pts_warped = np.around(np.vstack([pts_warped[1], pts_warped[0], np.ones_like(pts_warped[1])])).astype(int)

        # get the coordinates of points on the original image by applying an inverse transformation on pts_warped
        pts_original = np.around(np.matmul(np.linalg.inv(T), pts_warped)).astype(int)
        res[pts_warped[1], pts_warped[0], :] = im[pts_original[1], pts_original[0], :]

    return res

# produce a warp between im1 and im2 based on warp_frac (deciding shape) and dissolve_frac (deciding color)
def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    # make a copy of inputs to prevent mutation
    im1_pts_copy = im1_pts.copy()
    im2_pts_copy = im2_pts.copy()
    # decide the shape based on warp_frac
    morphed_pts =  (1 - warp_frac) * im1_pts_copy + warp_frac * im2_pts_copy

    # compute transformation matrices
    matrices1 = []
    matrices2 = []
    for t1, t2 in zip(im1_pts_copy[tri], morphed_pts.copy()[tri]):
        matrices1.append(computeAffine(t1, t2))
    for t1, t2 in zip(im2_pts_copy[tri], morphed_pts.copy()[tri]):
        matrices2.append(computeAffine(t1, t2))
    
    # apply the inverse warp on each image
    im1_warp = inverse_warp(im1, morphed_pts[tri], matrices1)
    im2_warp = inverse_warp(im2, morphed_pts[tri], matrices2)

    # decide the color based on dissolve_frac and return the result
    return (1 - dissolve_frac) * im1_warp +  dissolve_frac * im2_warp


class FaceKeypointTestDataset(Dataset):

    def __init__(self, images, boxes, transform=None):
        self.images_filename = images
        self.boxes = boxes
        self.transform = transform
        
    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = skio.imread(self.images_filename[idx], as_gray=True)
        box = self.boxes[idx]
        
        sample = {"image": image, "box": box}

        if self.transform:
            sample = self.transform(sample)

        return sample

class CroppingTest(object):
    def __call__(self, sample):
        image, box = sample["image"], sample["box"]
        im_h, im_w = image.shape[:2]
        left, top, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # handle negative box
        left = max(0, left)
        top = max(0, top)

        image_cropped = image[top:top+h, left:left+w]

        return {"image": image_cropped, "box": box}

class ResizeTest(object):
    def __init__(self, h, w):
        self.height = h
        self.width = w
    
    def __call__(self, sample):
        image,  box = sample["image"], sample["box"]
        h, w = image.shape[:2]

        image_resized = resize(image, (self.height, self.width))
        return {"image": image_resized, "box": box}


class NormalizeTest(object):
    def __call__(self, sample):
        image, box = sample["image"], sample["box"]
        image_normalized = image.astype(np.float32) / 255 - 0.5  
        return {"image": image_normalized, "box": box}

if __name__ == "__main__":
    # to prevent crash when running locally
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    my_img_filenames = ["./mypicture.jpg", "./myfriend.jpg"]
    bboxes = np.array([[120., 190., 250., 290.], [120., 180., 260., 300.]]).astype(np.float32)

    # crop and resize the images in my dataset
    ts = transforms.Compose([CroppingTest(), ResizeTest(224, 224), NormalizeTest()])
    my_dataset = FaceKeypointTestDataset(my_img_filenames, bboxes, transform=ts)

    # create dataloaders, using batch size of 2
    BATCH_SIZE = 2
    my_dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle = False)

    # upload trained model 
    state_dict = torch.load("./resnet18_for_part3.pt", map_location=torch.device('cpu'))
    resnet18 = models.resnet18(pretrained=True)
    resnet18.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    resnet18.fc = nn.Linear(512, 68 * 2, bias=True)
    resnet18.load_state_dict(state_dict)
    
    # get the keypoint set using the model
    sets = []
    with torch.no_grad():
        resnet18.eval()
        for sample_batched in my_dataloader:
            x = sample_batched["image"]
            x = x.unsqueeze(1)
            pred = resnet18(x)
            pred = torch.reshape(pred, (pred.shape[0], 68, 2))
            for j in range(pred.shape[0]):
                landmarks_pred = pred.detach()[j]
                # plot prediction keypoints on original image
                landmarks_origin = (landmarks_pred * 224).numpy()
                box = sample_batched["box"][j]
                im_origin = skio.imread(my_img_filenames[j])
                left = max(0, int(box[0]))
                top = max(0, int(box[1]))
                width = int(box[2])
                height = int(box[3])

                landmarks_origin[:, 0] *= width / 224
                landmarks_origin[:, 1] *= height / 224

                landmarks_origin += [left, top]
                landmarks_origin = np.array(landmarks_origin)
                sets.append(landmarks_origin)

    # add points to four corners
    im1 = skio.imread("./mypicture.jpg")
    im2 = skio.imread("./myfriend.jpg")
    h, w = im1.shape[:2]
    four_corners = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]])
    set1 = sets[0]
    set2 = sets[1]
    set1 = np.vstack([four_corners, set1])
    set2 = np.vstack([four_corners, set2])

    avg_pts = mean_points(set1, set2)
    tri = Delaunay(avg_pts)
    triangles = tri.simplices

    #plot the triangluation mesh on input images and save
    plt.imshow(im1)
    plt.triplot(set1[:, 0], set1[:, 1], triangles)
    plt.plot(set1[:, 0], set1[:, 1], 'o')
    plt.axis("off")
    plt.savefig("./results/triangulation1.jpg")
    plt.close()

    plt.imshow(im2)
    plt.triplot(set2[:, 0], set2[:, 1], triangles)
    plt.plot(set2[:, 0], set2[:, 1], 'o')
    plt.axis("off")
    plt.savefig("./results/triangulation2.jpg")
    plt.close()
    
    # generate 45 values evenly from [0, 1] for warp_frac and dissolve_frac
    N = 45
    factors = np.linspace(0.0, 1.0, N)

    
    # generate a sequence of warps 
    for i in range(factors.shape[0]):
        im_morphed = morph(im1, im2, set1, set2, triangles, factors[i], factors[i])
        skio.imsave("./results/gif/morphed" + str(i) + ".jpg", im_morphed)

    # generate a gif with the results above 
    images = []
    for i in range(N):
        images.append(imageio.imread("./results/gif/morphed" + str(i) + ".jpg"))

    imageio.mimsave('./results/me_and_myfriend.gif', images)