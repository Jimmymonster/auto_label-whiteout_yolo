from ultralytics import YOLO
import os
from PIL import Image, ImageDraw
import numpy as np
import shutil

def create_output_yolo_project(model, input_yolo_path, output_yolo_path, class_name, class_index, confidence_threshold=0.5, target_class_index=None):
    input_image_path = os.path.join(input_yolo_path, "images")
    input_label_path = os.path.join(input_yolo_path, "labels")
    input_class_path = os.path.join(input_yolo_path, "classes.txt")
    
    output_image_path = os.path.join(output_yolo_path, "images")
    output_label_path = os.path.join(output_yolo_path, "labels")
    output_class_path = os.path.join(output_yolo_path, "classes.txt")
    
    # Clear output path if it exists and recreate the directory structure
    if(input_yolo_path!=output_yolo_path):
        if os.path.exists(output_yolo_path):
            shutil.rmtree(output_yolo_path)
        os.makedirs(output_image_path)
        os.makedirs(output_label_path)
    
        # Copy the input classes.txt to the output
        shutil.copy(input_class_path, output_class_path)
    
    # Load and update classes.txt in the output project
    classes = []
    if os.path.exists(output_class_path):
        with open(output_class_path, 'r') as class_file:
            classes = class_file.read().splitlines()
    
    if class_name not in classes:
        with open(output_class_path, 'a') as class_file:
            class_file.write(f"{class_name}\n")
        classes.append(class_name)
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Iterate over each image
    for image_file in image_files:
        image_path = os.path.join(input_image_path, image_file)
        img = Image.open(image_path)
        img_width, img_height = img.size
        results = model(image_path)
        
        # Prepare the label data
        label_file_name = os.path.splitext(image_file)[0] + ".txt"
        input_label_file_path = os.path.join(input_label_path, label_file_name)
        output_label_file_path = os.path.join(output_label_path, label_file_name)
        
        existing_labels = []
        if os.path.exists(input_label_file_path):
            with open(input_label_file_path, 'r') as label_file:
                existing_labels = label_file.read().splitlines()
        
        with open(output_label_file_path, 'w') as label_file:
            # Write existing labels first
            for label in existing_labels:
                label_file.write(f"{label}\n")
            
            # Add new labels from model predictions in YOLO format
            for result in results:
                for box, conf, cls_idx in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    if conf >= confidence_threshold and (target_class_index is None or cls_idx == target_class_index):
                        x1, y1, x2, y2 = box[:4]
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        label_file.write(f"{class_index} {x_center} {y_center} {width} {height}\n")
        
        # Copy the image to the output images directory
        if(input_yolo_path!=output_yolo_path):
            output_image_file_path = os.path.join(output_image_path, image_file)
            shutil.copy(image_path, output_image_file_path)

def whiteout_prediction_areas(model, input_yolo_path, output_yolo_path, rectangle_color=(255, 255, 255), confidence_threshold=0.5, target_class_index=None):
    input_image_path = os.path.join(input_yolo_path, "images")
    input_label_path = os.path.join(input_yolo_path, "labels")
    input_class_path = os.path.join(input_yolo_path, "classes.txt")
    
    output_image_path = os.path.join(output_yolo_path, "images")
    output_label_path = os.path.join(output_yolo_path, "labels")
    output_class_path = os.path.join(output_yolo_path, "classes.txt")
    
    # Clear output path if it exists and recreate the directory structure
    if(input_yolo_path!=output_yolo_path):
        if os.path.exists(output_yolo_path):
            shutil.rmtree(output_yolo_path)
        os.makedirs(output_image_path)
        os.makedirs(output_label_path)
    
        # Copy the input classes.txt and labels to the output directory
        shutil.copy(input_class_path, output_class_path)
        for label_file in os.listdir(input_label_path):
            shutil.copy(os.path.join(input_label_path, label_file), os.path.join(output_label_path, label_file))
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Iterate over each image
    for image_file in image_files:
        image_path = os.path.join(input_image_path, image_file)
        results = model(image_path)
        
        # Open the image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Draw rectangles over the prediction areas
        for result in results:
            for box, conf, cls_idx in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if conf >= confidence_threshold and (target_class_index is None or cls_idx == target_class_index):
                    x1, y1, x2, y2 = map(int, box[:4])
                    draw.rectangle([x1, y1, x2, y2], fill=rectangle_color)
        
        # Save the modified image to the output directory
        output_image_file_path = os.path.join(output_image_path, image_file)
        img.save(output_image_file_path)
