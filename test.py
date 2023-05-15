import os
from time import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable as V

from networks.dplinknet import DPLinkNet34
from utils import get_patches, stitch_together

BATCHSIZE_PER_CARD = 2

# Check if a CUDA-enabled GPU is available otherwise use CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class TTAFrame():
    def __init__(self, net):
        # Initialize the network and use DataParallel if multiple GPUs are available
        self.net = net().to(device)
        if device == 'cuda':
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        else:
            self.net = torch.nn.DataParallel(self.net, device_ids=range(os.cpu_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        # Set the network to evaluation mode if required
        if evalmode:
            self.net.eval()

        # Determine the batch size based on the number of GPUs available
        if device == 'cuda':
            batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        else:
            batchsize = os.cpu_count() * BATCHSIZE_PER_CARD

        # Use different methods for TTA depending on the batch size
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        if device == 'cuda':
            img5 = V(torch.Tensor(img5).cuda())
        else:
            img5 = V(torch.Tensor(img5).cpu())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        if device == 'cuda':
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


DATA_NAME = "DIBCO"  # BickleyDiary, DIBCO, PLM
DEEP_NETWORK_NAME = DPLinkNet34  # LinkNet34, DLinkNet34, DPLinkNet34
solver = TTAFrame(DEEP_NETWORK_NAME)

img_indir = "uploaded_images"
img_outdir = "binarized_images"

if not os.path.exists(img_outdir):
    os.makedirs(img_outdir)

img_list = os.listdir(img_indir)
img_list.sort()

# print("Now loading the model weights:", f"weights/{DATA_NAME.lower()}_{DEEP_NETWORK_NAME.__name__.lower()}.th")
solver.load(f"weights/{DATA_NAME.lower()}_{DEEP_NETWORK_NAME.__name__.lower()}.th")

start_time = time()

# Iterate through each file in the input directory
for idx in range(len(img_list)):
    # Check if the current file is a directory, and skip it if it is
    if os.path.isdir(os.path.join(img_indir, img_list[idx])):
        continue

    # Extract the filename and extension of the current file, and create the input and output file paths
    fname, fext = os.path.splitext(img_list[idx])
    img_input = os.path.join(img_indir, img_list[idx])
    img_output = os.path.join(img_outdir, fname + "-" + DEEP_NETWORK_NAME.__name__ + ".tiff")

    # Load the image using OpenCV, resize it to 256x256, and save it as a PNG file
    img = cv2.imread(img_input)
    img = cv2.resize(img, (256, 256))
    cv2.imwrite('image.png', img)

    # Define the patch size and extract patches from the resized image
    TILE_SIZE = int(256)
    locations, patches = get_patches(img, TILE_SIZE, TILE_SIZE)

    # Initialize an empty list to store the predicted masks for each patch
    masks = []

    # Iterate through each patch and use the deep learning algorithm to predict the mask
    for idy in range(len(patches)):
        msk = solver.test_one_img_from_path(patches[idy])
        masks.append(msk)

    # Stitch together the predicted masks to create the final segmentation
    prediction = stitch_together(locations, masks, tuple(img.shape[0:2]), TILE_SIZE, TILE_SIZE)

    # Threshold the prediction to convert it to a binary mask, and save it as a TIFF file in the output directory
    prediction[prediction >= 5.0] = 255
    prediction[prediction < 5.0] = 0
    cv2.imwrite(img_output, prediction.astype(np.uint8))

print("Total running time: %f sec." % (time() - start_time))
print("Finished!")
