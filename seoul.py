from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import sys
from ultralytics import YOLO 
import cv2
import numpy as np
import torch
from torchvision import transforms

"""class yolov8_thread(QThread):
    def __init__(self, vision):
      super().__init__()
      self.vision = vision
      self.model = YOLO("yolov8n.pt")
      self.cap = cv2.VideoCapture(0)
      
    def run(self):
        while True:
            ret, frame = self.cap.read()
            result = self.model.predict(frame)
            predicted_image = result[0].plot()
            predicted_image = cv2.resize(predicted_image, (1920, 1080))
            predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
            predicted_image = QImage(predicted_image.data, predicted_image.shape[1], predicted_image.shape[0], QImage.Format_RGB888)
            self.vision.setPixmap(QPixmap.fromImage(predicted_image))
            self.vision.show()
      
class yolov8_main(QMainWindow):
    def __init__(self):
        super(yolov8_main, self).__init__()
        uic.loadUi("yolov8_ui.ui", self)
        self.show()

        self.t = yolov8_thread(self.vision)
        self.t.start()

app = QApplication(sys.argv)
windows = yolov8_main()
windows.show()
sys.exit(app.exec_())"""


cap = cv2.VideoCapture(0)          
model_path = "yolov8n.pt"
model = torch.load(model_path)

transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])



while True:
    ret, frame = cap.read()
    input_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(input_tensor)

        for prediction in predictions:
            x, y, w, h, confidence, class_idx = prediction
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            color = (0, 0, 255) 
            thickness = 2 
            image_cv2 = cv2.rectangle(image_cv2, (x1, y1), (x2, y2), color, thickness)
            cv2.imshow('d', image_cv2)
            cv2.waitKey(1)





"""
# 모델 가중치 로드
model_state_dict = torch.load("yolov8n.pt")

# 모델 구조 정의
model = YourModel()  # YourModel은 YOLOv8n 모델의 구조를 정의하는 클래스입니다.

# 모델에 가중치 로드
model.load_state_dict(model_state_dict)

# 모델을 평가 모드로 전환
model.eval()
scores = torch.randn(100) 

img = cv2.imread('image.png')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print("image shape",img.shape)

plt.figure(figsize=(8,8))

_, regions = selectivesearch.selective_search(img_rgb, scale=100, min_size=5000)



cand_rects = [cand['rect'] for cand in regions if cand['size'] > 10000] # rects만 가져와서 저장

green_rgb = (125,255,51)
img_rgb_copy = img_rgb.copy() # 그림 카피 뜨기

for rect in cand_rects:
    left=rect[0]
    top=rect[1]

    right = left + rect[2]
    bottom = top + rect[3]

    img_rgb_copy = cv2.rectangle(img_rgb_copy, (left,top),(right,bottom),color=green_rgb, thickness=2) 
    #사각형 좌표의 왼쪽 상단, 오른쪽 하단 입력


    # cv2.imshow로 이미지 출력
    #cv2.imshow("Candidate Regions", img_rgb_copy)
    #cv2.waitKey(0)  # 키 입력 대기
    #cv2.destroyAllWindows()  # 창 닫기

    # candidate regions에서 숫자 영역 추출
    digit_region = img_rgb_copy[rect[1]:rect[3], rect[0]:rect[2]]

    # 숫자 영역을 28x28 크기로 변환
    digit_region = cv2.resize(digit_region, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

    # 숫자 영역을 텐서로 변환
    digit_tensor = torch.from_numpy(np.array(digit_region)).float()

    # 숫자 분류
    prediction = model(digit_tensor).argmax(dim=1)
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tvmonitor"]
    prediction_class = class_names[prediction]

    print(prediction_class)

    # 숫자 출력
    print(prediction)"""