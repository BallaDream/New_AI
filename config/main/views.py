from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import os
import cv2
import torch
import gc
from PIL import Image
from ultralytics import YOLO
import numpy as np
import pandas as pd
import torch.nn as nn
import uuid
import torchvision.transforms as transforms
import torchvision.models as models
import json
from django.http import JsonResponse
device = "cpu" if torch.cuda.is_available() else "device"
Face_eye_detection_model = YOLO("main/best_weight_model/newfaceeye.pt").to(device)
Object_detection_model = YOLO("main/best_weight_model/face_segmentation.pt").to(device)


class WrinkleRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 6.0  # 등급 범위에 맞게 조절
        return x
class PigmentationRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 5.0  # 등급 범위에 맞게 조절
        return x
class PoreRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 5.0  # 등급 범위에 맞게 조절
        return x
class DrynessRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 4.0  # 등급 범위에 맞게 조절
        return x
class SaggingRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 6.0  # 등급 범위에 맞게 조절
        return x

class RegressionEfficientNet(nn.Module):
    def __init__(self,label):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid() 
        self.label = label
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        if self.label =='Wrinkle':
            x = self.act(x) * 6.0  # 등급 범위에 맞게 조절
        elif self.label =='Pore':
            x = self.act(x) * 5.0  # 등급 범위에 맞게 조절
        elif self.label =='Dry':
            x = self.act(x) * 4.0  # 등급 범위에 맞게 조절
        elif self.label =='Sagging':
            x = self.act(x) * 6.0  # 등급 범위에 맞게 조절
        elif self.label =='Pigmentation':
            x = self.act(x) * 7.0  # 등급 범위에 맞게 조절
        return x
wrinkle_model = RegressionEfficientNet('Wrinkle').to(device)
wrinkle_model.load_state_dict(torch.load("main/best_weight_model/best_wrinkle_model.pth", map_location=device))
wrinkle_model.eval()
Pore_model = RegressionEfficientNet('Pore').to(device)
Pore_model.load_state_dict(torch.load("main/best_weight_model/Pore_best_model_2nd.pth", map_location=device))
Pore_model.eval()
Pig_model = RegressionEfficientNet('Pigmentation').to(device)
Pig_model.load_state_dict(torch.load("main/best_weight_model/Pigmentation_best_model_2nd.pth", map_location=device))
Pig_model.eval()
Sagging_model = RegressionEfficientNet('Sagging').to(device)
Sagging_model.load_state_dict(torch.load("main/best_weight_model/Sagging_best_model_2nd.pth", map_location=device))
Sagging_model.eval()
Dry_model = RegressionEfficientNet('Dry').to(device)
Dry_model.load_state_dict(torch.load("main/best_weight_model/Dry_best_model_2nd.pth", map_location=device))
Dry_model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
part_models = {
    'forehead': [("Wrinkle", wrinkle_model), ("Pigmentation", Pig_model)],
    'glabella': [("Wrinkle", wrinkle_model)],
    'l_periocular': [("Wrinkle", wrinkle_model)],
    'r_periocular': [("Wrinkle", wrinkle_model)],
    'l_cheek': [("Pore", Pore_model), ("Pigmentation", Pig_model)],
    'r_cheek': [("Pore", Pore_model), ("Pigmentation", Pig_model)],
    'lips': [("Dry", Dry_model)],
    'chin': [("Sagging", Sagging_model)]
}
def predict_score(crop, model):
    print("predict")
    input_tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
    print("predict_fin")
    return prediction.item()
def convert_results(results_by_part: dict) -> dict:
    output = {}
    print("convert")
    part_name_map = {
        "forehead": "Forehead",
        "glabella": "Glabella",
        "l_periocular": "LeftEye",
        "r_periocular": "RightEye",
        "l_cheek": "LeftCheek",
        "r_cheek": "RightCheek",
        "lips": "Lips",
        "chin": "Jawline",
    }

    issue_key_map = {
        "Wrinkle": "wrinkle",
        "Pigmentation": "pigment",
        "Pore": "pore",
        "Dry": "dry",
        "Sagging": "elastic",
    }

    suffix_map = {
        "Dry": "Score",
        "Sagging": "SaggingScore",
        "Wrinkle": "Score",
        "Pigmentation": "Score",
        "Pore": "Score",
    }

    for part, issues in results_by_part.items():
        for issue_type, raw_score in issues.items():
            key_prefix = issue_key_map.get(issue_type)
            part_label = part_name_map.get(part)
            suffix = suffix_map.get(issue_type, "Score")
            
            if key_prefix and part_label:
                key = f"{key_prefix}{part_label}{suffix}"
                output[key] = round(raw_score)
    print("convert_fin")
    return output

def index(request):
    return render(request,"main/index.html")
@csrf_exempt    
def analyze(request):
    img_id = str(uuid.uuid4())
    input_path = f"main/static/input_{img_id}.jpg"
    output_path = f"main/static/output_{img_id}.jpg"
    img_path = input_path
    file = request.FILES['file']
    with open(img_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    results = Face_eye_detection_model(img_path)
    boxes = results[0].boxes
    cls = boxes.cls.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    face_indices = [i for i, (c, cf) in enumerate(zip(cls, conf)) if c == 1 and cf >= 0.7]
    eye_indices = [i for i, c in enumerate(cls) if c == 0]

    img = cv2.imread(img_path)
    result_msg = ""
    show_image = False
    status =200
    if face_indices:
        face_boxes = [xyxy[i] for i in face_indices]
        if len(face_boxes) == 1:
            if eye_indices:
                eye_boxes = [xyxy[i] for i in eye_indices]
                if len(eye_boxes) == 2:
                    eye_boxes_sorted = sorted(eye_boxes, key=lambda box: box[0])
                    left_eye, right_eye = eye_boxes_sorted[0], eye_boxes_sorted[-1]

                    start_point = (int(left_eye[0]), int(left_eye[1]))
                    end_point = (int(right_eye[2]), int(right_eye[3]))
                    cv2.rectangle(img, start_point, end_point, (0, 0, 0), thickness=-1)

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)

                    # results2 = model2(pil_img)
                    results2 = Object_detection_model(pil_img)
                    
                    if results2[0].boxes is not None and len(results2[0].boxes) > 0:
                        results2[0].save(filename=output_path)
                        
                        boxes2 = results2[0].boxes

                        cls2 = boxes2.cls.cpu().numpy()
                        xyxy2 = boxes2.xyxy.cpu().numpy()

                        index_mapping = {
                                'forehead': [i for i, c in enumerate(cls2) if c == 1],
                                'glabella': [i for i, c in enumerate(cls2) if c == 2],
                                'l_periocular': [i for i, c in enumerate(cls2) if c == 3],
                                'r_periocular': [i for i, c in enumerate(cls2) if c == 4],
                                'l_cheek': [i for i, c in enumerate(cls2) if c == 5],
                                'r_cheek': [i for i, c in enumerate(cls2) if c == 6],
                                'lips': [i for i, c in enumerate(cls2) if c == 7],
                                'chin': [i for i, c in enumerate(cls2) if c == 8],
                        }

                        results_by_part = {}
                         # === 각 부위별로 crop 및 예측 수행 ===
                        for part, indices in index_mapping.items():
                            if not indices:
                                continue  # 해당 부위가 탐지되지 않으면 skip
    
                                # 첫 번째 탐지된 부위 기준으로 crop
                            x1, y1, x2, y2 = map(int, xyxy2[indices[0]])
                            crop = pil_img.crop((x1, y1, x2, y2))
    
                            part_results = {}
                            for label, model in part_models.get(part, []):
                                score = predict_score(crop, model)
                                part_results[label] = round(score, 2)
    
                            if part_results:
                                results_by_part[part] = part_results
                        print(index_mapping)        
                        final_result = convert_results(results_by_part)
                        result_msg = final_result
                        show_image = True
                        if all(len(indices) == 0 for indices in index_mapping.values()):
                            result_msg= "얼굴 부위 식별 불가능합니다. 이미지를 다시 선택해주세요"
                            status = 422
                    else:
                        result_msg = "유의미한 결과가 없습니다. 이미지를 다시 선택해주세요."
                        status = 422
                        final_result = result_msg
                else:
                    result_msg = "눈이 2개 이상 탐지되지 않았습니다. 다시 사진을 선택해주세요."
                    status = 422
                    final_result = result_msg
            else:
                result_msg = "눈이 탐지되지 않았습니다. 다시 사진을 선택해주세요."
                status = 422
                final_result = result_msg
        else:
            result_msg = "얼굴이 2개 이상 탐지되었습니다. 다시 사진을 선택해주세요."
            status = 422
            final_result = result_msg
    else:
        result_msg = "얼굴이 탐지되지 않았습니다. 다시 사진을 선택해주세요."
        status = 422
        final_result = result_msg

    # return JsonResponse(final_result,status= status)
    return JsonResponse({
    "result": final_result,
    "image_url": output_path if show_image else ""
}, status=status)
# Create your views here.
    