import numpy as np
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import sys
from  PIL import Image
from tqdm import tqdm

sys.path.append('../..')
sys.path.append('../../PointPillars')
sys.path.append('../../PointPillars.ops')
from MLP_model import MLPModel
from Dataloader import Dataset
from ultralytics import YOLO
from PointPillars.test import *
from MLP_utils import *

def main(args):
    KITTI_PATH = args.kitti_path
    CKP_PATH = args.ckp_path
    VAL_PATH = args.val_path
    RESULT_PATH = args.result_path


    data_loader_train = Dataset(KITTI_PATH + '/velodyne_reduced', KITTI_PATH + '/calib',
                                KITTI_PATH + '/image_2', KITTI_PATH + '/label_2', val_image_ids=VAL_PATH)

    model=MLPModel()
    checkpoint = torch.load(CKP_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    YOLO_model = YOLO(args.yolo_path)
    print('Training epoch {}, training loss'.format(checkpoint['epoch'], checkpoint['loss']))

    model.eval()
    for idx,((point_file, calib_file), img_file, gt_bbox) in enumerate(tqdm(data_loader_train, desc='Data no.')):
        if VAL_PATH:
            idx = data_loader_train.val_image_ids[idx]
        img_w, img_h = Image.open(img_file).size
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
            YOLO_results = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.double)
        '''
        PointPillars predict
        '''
        PointPillars_results = PointPillars_test(point_file, calib_file, img_file,
                                                ckpt='/home/server1/Wu/Multi-sensory-fusion/PointPillars/pretrained/epoch_160.pth')
        if len(PointPillars_results) == 0:
            MLP_results = []
        else:
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
            '''
            Predict
            '''
            predicts = []
            scores = torch.tensor([])
            with torch.no_grad():
                for box_idx, single_input in enumerate(input):
                    predict = model(single_input)
                    predict, score = torch.split(predict, [4, 1])
                    scores = torch.cat([scores, score], dim=0)
                    predicts.append(predict)
            scores = torch.sigmoid(scores).numpy()
            predicts = torch.stack(predicts).numpy()

            enclosing_box = enclosing_box[:,:-1].numpy()
            enclosing_box_width = enclosing_box[:,2] - enclosing_box[:,0]
            enclosing_box_height = enclosing_box[:,3] - enclosing_box[:,1]
            enclosing_box_w_h = np.concatenate((enclosing_box_width[np.newaxis,:].T, enclosing_box_height[np.newaxis,:].T),axis=1)

            predict_w_h = enclosing_box_w_h * 2 * predicts[:,2:]
            predict_center = enclosing_box[:,:2] + enclosing_box_w_h * predicts[:,0:2]
            predict_box = np.concatenate([predict_center - predict_w_h / 2, predict_center + predict_w_h / 2],axis=1)

            MLP_results = np.concatenate([predict_box,scores[np.newaxis,:].T],axis=1)


        with open(RESULT_PATH + f'{str(idx).zfill(6)}.txt', 'w') as f:
            # if len(YOLO_results) == 0:
            #     f.write(' '.join(['DontCare',' '.join(['0' for i in range(15)])]))
            for i in range(len(MLP_results)):
                useless1 = '0.00 0 0.00'
                useless2 = ' '.join(['0.00' for i in range(7)])
                content = ' '.join(['Pedestrian', useless1, ' '.join(['{:.2f}'.format(i) for i in MLP_results[i][:-1]]),
                                    useless2, '{:.2f}'.format(MLP_results[i][-1])])
                print(content)
                f.write(content + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--kitti_path', default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--ckp_path', default='../../BBP_0.7_GT_0.5.pth', help='path of fusion model checkpoints')
    parser.add_argument('--val_path', default='../../ImageSets/val.txt', help='path of KITTI val.txt')
    parser.add_argument('--result_path', default='../results/MLP_results/', help='path to save your results')
    parser.add_argument('--yolo_path', default='../../yolov8n.pt', help='path of yolo ckp')
    
    args = parser.parse_args()

    main(args)