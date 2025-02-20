'''
:Author: Yuhong Wu
:Date: 2023-12-02 21:44:31
:LastEditors: Yuhong Wu
:LastEditTime: 2023-12-15 20:14:55
:Description: 
'''
import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import time
from MLP_model import MLPModel
from Dataloader import Dataset
from ultralytics import YOLO
from MLP_utils import *
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append('PointPillars')
sys.path.append('PointPillars.ops')
from PointPillars.test import *

torch.manual_seed(3407)
EPOCH = 200
KITTI_TRAIN_PATH = '/media/server1/5150/Wu/KITTI/training'

ciou_loss = CIOULoss()
# BCE_loss = torch.nn.BCEWithLogitsLoss()
BCE_loss = torch.nn.BCELoss()
data_loader_train = Dataset(KITTI_TRAIN_PATH+'/velodyne_reduced',KITTI_TRAIN_PATH+'/calib',
                            KITTI_TRAIN_PATH+'/image_2',KITTI_TRAIN_PATH+'/label_2')
model=MLPModel()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
YOLO_model = YOLO("yolov8n.pt")
writer = SummaryWriter()

model.train()
for epoch in tqdm(range(EPOCH), desc='Epoch'):
    sum_loss=0
    sum_loss_regression = 0
    sum_loss_conf = 0
    idx = 0
    scheduler.step()
    for (point_file, calib_file), img_file, gt_boxes in tqdm(data_loader_train, desc='Data no.'):
        gt_boxes = torch.tensor(gt_boxes)
        img_w, img_h = Image.open(img_file).size
        if len(gt_boxes) == 0:
            continue

        '''
        YOLO predict
        '''
        YOLO_boxes = YOLO_model.predict(img_file,classes=0)[0].boxes
        YOLO_results = []
        for box in YOLO_boxes:
            if box.cls == 0:
                YOLO_results.append(torch.concat([box.xyxy.squeeze(), box.conf], dim=0).tolist())
        YOLO_results = torch.tensor(YOLO_results)
        # print('YOLO result:',YOLO_results)
        if len(YOLO_results) == 0:
            # YOLO_results = torch.tensor([[-1,-1,-1,-1,-1]])
            YOLO_results = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.double)

        '''
        PointPillars predict
        '''
        PointPillars_results = PointPillars_test(point_file, calib_file, img_file,
                                                 ckpt='/home/server1/Wu/Multi-sensory-fusion/PointPillars/pretrained/epoch_160.pth')
        if len(PointPillars_results) == 0:
            continue
        # print('PointPillars result:', PointPillars_results)
        '''
        Normalize
        '''
        YOLO_results /= torch.tensor([img_w, img_h, img_w, img_h, 1])
        PointPillars_results /= torch.tensor([img_w, img_h, img_w, img_h, 1])

        '''
        Bounding Box Pair
        '''
        input, enclosing_box = align_boxes(YOLO_results, PointPillars_results)
        enclosing_box *= torch.tensor([img_w, img_h, img_w, img_h, 1])
        # print('input:',input)
        # print('enclosing_box:', enclosing_box)
        '''
        Predict and compute loss
        '''
        loss = torch.autograd.Variable(torch.Tensor([0]))
        scores = torch.tensor([])
        gt_scores = torch.tensor([])
        for box_idx, single_input in enumerate(input):
            predict = model(single_input)
            predict, score = torch.split(predict,[4,1])
            scores = torch.cat([scores,score],dim=0)
            optimizer.zero_grad()
            single_regression_loss, gt_score = ciou_loss(enclosing_box[box_idx], predict, gt_boxes)
            gt_scores = torch.cat([gt_scores, gt_score], dim=0)
            loss = loss + 50 * single_regression_loss
        loss_regression = loss
        loss_conf = 50 * BCE_loss(scores, gt_scores)
        loss = loss + loss_conf
        print(loss_conf)
        print(loss_regression)
        loss = loss / len(input)
        if loss == 0:
            continue

        loss.backward()
        # for name, para in model.linear4.named_parameters():
        #     print('-->name:', name)
        #     print('-->para:', para)
        #     print('-->grad_requires:', para.requires_grad)
        #     print('-->grad:', para.grad)
        optimizer.step()
 
        sum_loss+=loss.data
        sum_loss_regression+=loss_regression.data/ len(input)
        sum_loss_conf+=loss_conf.data/ len(input)
        idx += 1
    print('epoch[%d/%d] loss:%.03f' % (epoch + 1, EPOCH, sum_loss / idx))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    writer.add_scalar('loss', sum_loss / idx, epoch)
    writer.add_scalar('loss_regression', sum_loss_regression / idx, epoch)
    writer.add_scalar('loss_conf', sum_loss_conf / idx, epoch)
    
    if epoch % 3 == 0:
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': sum_loss / idx}, f'{epoch}.pth')
