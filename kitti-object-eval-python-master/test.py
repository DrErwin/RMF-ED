'''
:Author: Yuhong Wu
:Date: 2024-06-16 20:27:33
:LastEditors: Yuhong Wu
:LastEditTime: 2025-02-21 00:17:03
:Description: 
'''
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
det_path = r"D:\Multi-sensor_Fusion\code\experiment\results\PointPillars_results"
dt_annos = kitti.get_label_annos(det_path)
gt_path = r"D:\Multi-sensor_Fusion\Dataset\label_2"
gt_split_file = r"D:\Multi-sensor_Fusion\code\ImageSets\val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer
# print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer