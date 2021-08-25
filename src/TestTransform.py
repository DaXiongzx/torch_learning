from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

imgPath = "data/train/ants_image/0013035.jpg"
imgAbsPath = "F:\torch_learning\data\train\ants_image\0013035.jpg"
cvImg = cv2.imread(imgPath) #use cv so cvImg is type of np.ndarry narrays矩阵
img = Image.open(imgPath) #PIL image
writer = SummaryWriter("logs")
# 1 how to use tansforms(python)
tensorTrans = transforms.ToTensor() #ToTensor is a class not a method so tensorTrans is an objective and it has no args in constructive
tensorImg = tensorTrans(img) #method __call__: return tensor image / args:PIL image
#transform PIL image to torch.tensor

writer.add_image("TensorImage",tensorImg) #args:torch.tensor or np.array
writer.close()