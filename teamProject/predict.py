import numpy as np, cv2
import tensorflow as tf
import mediapipe as mp
from utils import *

face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0)
model = tf.keras.models.load_model("./model")  #저장된 모델 불러오기
capture = cv2.VideoCapture(0)

class_name = ["CJY", "LKS", "LSK"]
img_width, img_height = 180, 180

while True:
  success, image = capture.read()
  if not success: continue
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  detection_results = face_detection.process(image)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  if detection_results.detections :
    for detection in detection_results.detections:
      location_data = detection.location_data
      if location_data.format == location_data.RELATIVE_BOUNDING_BOX:
        bounding_box = location_data.relative_bounding_box
        x,y,w,h = int(bounding_box.xmin*image.shape[1]), int(bounding_box.ymin*image.shape[0]), \
                  int(bounding_box.width*image.shape[1]), int(bounding_box.height*image.shape[0])
        cv2.rectangle(image, (x, y, w, h), (0, 0, 255), 3)
        copy_image = image[y:y+h,x:x+w]
        copy_image = cv2.resize(copy_image, (img_height,img_width))
        copy_image_array = tf.expand_dims(copy_image, 0)
        predict = model.predict(copy_image_array)
        score = tf.nn.softmax(predict[0])

        if class_name[np.argmax(score)] == class_name[0] or class_name[np.argmax(score)] == class_name[2]:
          cv2.putText(image,"Unlock",(0,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0))
          cv2.putText(image,class_name[np.argmax(score)],(0,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
          cv2.imshow("predict", image)
          cv2.imshow("test", copy_image)
        else:
          cv2.putText(image, "Lock", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
          cv2.putText(image, class_name[np.argmax(score)], (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
          cv2.imshow("predict", image)
          cv2.imshow("test", copy_image)
  else:
    cv2.imshow("predict", image)
  if cv2.waitKey(10)==ord(" "):
    break
capture.release()