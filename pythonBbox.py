from ultralytics import YOLO
import os
import time
import torch

start = time.time()

device="cuda" if torch.cuda.is_available else "cpu"
model = YOLO("yolov8n.pt").to(device)

# YOLO 모델 로드
model = YOLO("yolov8n.pt").to(device)

# 이미지 파일이 있는 폴더 경로
image_folder = "./image"

# 정답 파일이 있는 폴더 경로
gt_folder = "./labels"

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 유사한 클래스 매핑 딕셔너리
class_mapping = {
    '0' : ['Pedestrian', 'Person', 'Cyclist'], # person
    '1' : ['Cyclist'], # bicycle
    '2' : ['Car', 'Van'], # car
    '3' : ['Cyclist'], # motocycle
    '5' : ['Car', 'Van'], # bus
    '7' : ['Truck'] # truck
}

target_classes = [0,1,2,3,5,7]  # classes to predict
gt_list = ['Pedestrian', 'Person', 'Cyclist', 'Car', 'Van', 'Truck']

# 탐지 클래스가 매핑 테이블에 있는지 확인
def is_similar_class(det_class, gt_class):
    if det_class in class_mapping:
        return gt_class in class_mapping[det_class]
    # 매핑되지 않은 클래스는 정확히 일치해야 함
    return det_class == gt_class

# 정답 파일 로드 함수
def load_ground_truth(gt_path):
    ground_truths = []
    with open(gt_path, 'r') as f:
        for line in f:
            li = line.split()
            class_id, x1, y1, x2, y2 = str(li[0]), float(li[4]), float(li[5]), float(li[6]), float(li[7])
            if class_id in gt_list:
                ground_truths.append([class_id, x1, y1, x2, y2])
    return ground_truths

# IoU 계산 함수
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

# 평가 함수, precision recall avg_iou 를 반환
def evaluate_detections(detections, ground_truths, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truths)        # 검출해야 할 것(gt)의 총 개수
    ious = []

    for det in detections:
        detected = False
        
        for gt in ground_truths:
            iou = calculate_iou(det[1], gt[1:])
        
            if iou >= iou_threshold and is_similar_class(str(det[0]), gt[0]):   # IoU가 0.5 이상이고 클래스가 일치할 경우, 즉 검출해야할 것이었음
                true_positives += 1     # 예측 성공
                detected = True
                false_negatives -= 1        # 검출에 성공했으므로 검출할 것을 미검출한거(FN)에 대해 -1
                ious.append(iou)
                ground_truths.remove(gt)
                break

        if not detected:    # det가 있는 시점에서 검출은 한건데, 검출하면 안될 것을 검출했음
            false_positives += 1
        

    # 조건에 해당하지 않으면 0을 할당
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    avg_iou = sum(ious) / len(ious) if ious else 0


    return precision, recall, avg_iou

# 결과 누적 변수
total_precision = 0
total_recall = 0
total_iou = 0
num_images = 0

# 이미지 파일들에 대해 평가 수행
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    gt_path = os.path.join(gt_folder, os.path.splitext(image_file)[0] + '.txt')

    # 정답 로드
    ground_truths = load_ground_truth(gt_path)

    # 예측 수행
    results = model.predict(image_path, conf=0.4, device=device, save=False, classes=target_classes)

    # 탐지 결과 정리
    detections = []
    for result in results:
        boxes = results[0].boxes.xyxy  # x1, y1, x2, y2
        confs = results[0].boxes.conf  # confidence
        cls = results[0].boxes.cls  # class

        for box, conf, cl in zip(boxes, confs, cls):
            detections.append([int(cl), box.tolist()])

    # 평가
    precision, recall, avg_iou = evaluate_detections(detections, ground_truths)

    # 결과 누적
    if precision != 0 and recall !=0 and avg_iou !=0:
        total_precision += precision
        total_recall += recall
        total_iou += avg_iou
        num_images += 1

    print(f"Results for {image_file} - Precision: {precision:.4f}, Recall: {recall:.4f}, Avg IoU: {avg_iou:.4f}")

# 전체 평균 계산
avg_precision = total_precision / num_images
avg_recall = total_recall / num_images
avg_iou = total_iou / num_images

print("---"*10)
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average IoU: {avg_iou:.4f}")
print(f"Predicted Img: {num_images}")
print(f"Total Precision: {total_precision}")

end = time.time()

print(f"Proccess time:{end - start:.5f} sec")
