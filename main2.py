from ultralytics import YOLO
import os
import shutil
from utils import create_output_yolo_project,whiteout_prediction_areas,get_folders_list


many_folder_path = "C:/Users/thanapob/Downloads/TNN_Test"
folders = get_folders_list(many_folder_path)

model_path = "model/tnn5classes.pt"
confident_level = 0.2

class_name = "TNN16Live"
output_class_index = 1
target_class_index = [39,1]

rectangle_color = (255,255,255)

model = YOLO(model_path)

for folder in folders:

    # create_output_yolo_project(model, folder, folder, class_name, output_class_index, confidence_threshold=confident_level, target_class_index = target_class_index)

    whiteout_prediction_areas(model, folder, folder, rectangle_color, confidence_threshold=confident_level, target_class_index=target_class_index)