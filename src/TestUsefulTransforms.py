from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
imgPath = "data/train/ants_image/0013035.jpg"
img = Image.open(imgPath)
print(type(img))
#ToTensor 的使用，以PIL image 为参数 构成tensor
trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img)
writer.add_image("ToTensor",img_tensor) #img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data


#Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

#Resize
print(img.size)
trans_resize = transforms.Resize((256,256))
img_resize = trans_resize(img)
print("------------")
print(img_resize)
img_resize = trans_to_tensor(img_resize) #由PIL转成tensor
print("------------")
print(img_resize)
writer.add_image("resize",img_resize,0)
print(img_resize)

#compose是一套组合操作
trans_resize_2 = transforms.Resize(512) #都是 transforms类型。trans_to_tensor的输入是以trans_resize_2的输出来的 即类型相同
# ImgResize是由PIL->PIL   trans_to_tensor是由PIL->tensor  中间都是PIL
trans_compose = transforms.Compose([trans_resize_2,trans_to_tensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize",img_resize_2,1)

trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random,trans_to_tensor])
for i in range(10):
    img_crop =trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()
