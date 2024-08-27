from ultralytics import YOLO
import os
import shutil
from utils import create_output_yolo_project,whiteout_prediction_areas

input_path = "project"
output_path = "output"
model_path = "model/tnn5classes.pt"
confident_level = 0.2


target_class_index = [29,39]
output_class_index = {29:3, 39:1}
rectangle_color = (255,255,255)

#clear output path
# if(input_path!=output_path):
#     if os.path.exists(output_path):
#         shutil.rmtree(output_path)
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

model = YOLO(model_path)

create_output_yolo_project(model, input_path, output_path, output_class_index, confidence_threshold=confident_level, target_class_index = target_class_index)

# whiteout_prediction_areas(model, input_path, output_path, rectangle_color, confidence_threshold=confident_level, target_class_index=target_class_index)