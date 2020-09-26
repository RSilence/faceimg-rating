import sys, torch, random
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append("/home/rilea/exp0519/tuxiangchuli/task4")

 
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib as plt

from torchvision.models import resnet50
from train.create_datasetforL import faceDataset
import torch.nn as nn


with torch.no_grad():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                transforms.Lambda(lambda x: x.repeat(1,1,1))]) # 由于图片是单通道的，所以重叠三张图像，获得一个三通道的数据
    }



    test_img_dir = "/home/rilea/exp0519/tuxiangchuli/task4/data/imgali/img_align_celeba"
    test_rating_path = "/home/rilea/exp0519/tuxiangchuli/task4/data/Rating_Collection/test_rating.csv"

    test_dataset = faceDataset(test_img_dir,  data_transform["val"],  train_or_test="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # dataiter = iter(testloader)
    # img_name, images = dataiter.next()

    # print('GroundTruth: ', ' '.join('%5s' % img_name[j] for j in range(8)))
    
 
    PATH = './netargs/face_net__meandata_cpu_bs4_epoch10_Adam_pretrain_50605.pth'

    net = resnet50()
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, 1)
    net.load_state_dict(torch.load(PATH))
    net.eval() 
    
    f = open("/home/rilea/exp0519/tuxiangchuli/task4/data/imgali/ratings.txt","w")
    fclean = open("/home/rilea/exp0519/tuxiangchuli/task4/data/imgali/ratings_clean.txt","w")
    for i, data in enumerate(testloader, 0):
        img_name, img = data
        outputs = net(img)
        outputs = torch.squeeze(outputs.view(1,-1))
        n8 = outputs.numpy()
        # print("outputs:", n8)
        # print(img_name)
        for output,idxx in zip(n8,range(8)):
            f.writelines("{} {}\n".format(img_name[idxx],output/1400*100))
            fclean.writelines("{}\n".format(output/1400*100))
        print(img_name)

    f.close()
    fclean.close()
    # outputs = net(images)
    # print(outputs)
