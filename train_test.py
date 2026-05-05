import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from losses import *
import shutil

def train_val():
    output_folder=r'D:/biyesheji/PyTorch-Neural-Image-Assessment-master/test'
    image_folder = r'D:/biyesheji/PyTorch-Neural-Image-Assessment-master/image'# <- 图片目录
    anno_file = r'D:/biyesheji/PyTorch-Neural-Image-Assessment-master/AVA.txt'# <- AVA.txt 的绝对路径
    judgeNet = model.load_MobileNetV2_judge(pretrained=True).cuda()#model.JudgeMcJudgeFace().cuda()
    ## Loading Pre-Trained MobileNetV2 weights.
    #print(judgeNet.state_dict().keys())


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    ## Setting up Preprocessing Parameters
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize])
    
    seed=0    
    num=0 ##K 折交叉验证中的第 num 折,num可取0,1,2,3,4



    
    

    trainDataFeeder = dataloader.imageAssessmentLoader(image_folder, anno_file, train_transform,num,'train',seed=seed)
    testDataFeeder  = dataloader.imageAssessmentLoader(image_folder, anno_file, test_transform,num,'test',seed=seed)

    train_loader = torch.utils.data.DataLoader(
    trainDataFeeder, batch_size=64, shuffle=True,
    num_workers=2, pin_memory=True, collate_fn=dataloader.custom_collate_fn)

    test_loader = torch.utils.data.DataLoader(
    testDataFeeder, batch_size=64, shuffle=False,
    num_workers=2, pin_memory=True, collate_fn=dataloader.custom_collate_fn)
    optimizer = torch.optim.SGD(judgeNet.parameters(), lr=0.003)

    ##自己写的内容
    test_ID=list(testDataFeeder.dataDict.keys())
    ##test_Score=list(testDataFeeder.dataDict.values)
    for ID in test_ID:
        ID_folder=image_folder+"/"+ID+".jpg"
        output_folder_ID=output_folder+"/image"
        output_folder_score=output_folder+"/score"
        try:
            shutil.copy2(ID_folder, output_folder_ID)
        except FileNotFoundError:
            print(f"File {ID_folder} not found. Skipping copy.")
            continue
        arr = np.array(testDataFeeder.dataDict[ID], dtype=np.float32)
        arr = arr / arr.sum()
        np.savetxt(output_folder_score + "/" + ID + ".txt", arr, fmt="%.18e")

    
    for epoch in range(5):
        print("这是第epoch轮",epoch)
        for i, (images, scores) in enumerate(train_loader):
            print("这是第i次迭代",i)
            images = images.cuda()
            scores = scores.cuda().float().unsqueeze(2)

            predicted_dist = judgeNet(images).view(-1,10,1)

            loss = earth_mover_distance_loss_batch(predicted_dist, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print("Epoch:", epoch, "Iter:", i+1, '/', len(train_loader))
                print("EMD Loss:", loss.item())
            if (i+1) % 100 == 0:
                os.makedirs('D:/biyesheji/PyTorch-Neural-Image-Assessment-master/result', exist_ok=True)
                for sample in range(5):
                    txtData = np.savetxt('D:/biyesheji/PyTorch-Neural-Image-Assessment-master/result/res' + str(i+1) + '_' + str(epoch) + '_' + str(sample) +'_'+str(num)+'.txt', predicted_dist[sample].squeeze().data.cpu().numpy())
                    torchvision.utils.save_image(images[sample], 'D:/biyesheji/PyTorch-Neural-Image-Assessment-master/result/'+str(i+1)+ '_' + str(epoch)  + '_' + str(sample) +'_'+str(num)+'.jpg', normalize=True)
        os.makedirs('D:/biyesheji/PyTorch-Neural-Image-Assessment-master/result/snapshots', exist_ok=True)
        torch.save(judgeNet.state_dict(), 'D:/biyesheji/PyTorch-Neural-Image-Assessment-master/result/snapshots/' + "Epoch" + str(epoch) +'_'+str(num)+'.pth')
    







if __name__ == '__main__':

    train_val()


