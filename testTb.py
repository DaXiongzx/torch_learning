from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer = SummaryWriter("logs")
imgPath = "dataset/train/bees/16838648_415acd9e3f.jpg"
imgPil = Image.open(imgPath)
imgArray = np.array(imgPil)

writer.add_image("train",imgArray,1,dataformats="HWC")
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()