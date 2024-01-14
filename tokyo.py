import tensorflow as tf 
import selectivesearch
import numpy as np 
import cv2

model = tf.keras.models.load_model('mnist.h5')
number_image = cv2.imread("number.png")
number_image = cv2.resize(number_image, (640, 480))

_, regions = selectivesearch.selective_search(number_image, scale=5000, min_size=500)

#(x1, y1, width1, height1)

for cand in regions:
    try:
        length = cand['size']
        if length < 50000 and length > 100:
            rect = cand['rect']
            print(cand['size'])
            print("aa : {}, {}. {}. {} : ".format(rect[0], rect[1], rect[2], rect[3]))
  
            x1 =rect[0]
            y1 =rect[1]
            width = rect[2]
            height = rect[3]

            digit_region = number_image[int(y1):int(y1+height), int(x1):int(x1+width)]
            img = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28))
            cv2.imshow("Candidate Regions", img)
            x = img.reshape(1, 28, 28, 1)
            pred = model.predict(x)
            print(np.argmax(pred[0]))
            print(pred)

            cv2_image = cv2.rectangle(number_image, (int(x1), int(y1)), (int(x1+width), int(y1+height)), color=(255, 0, 0), thickness=2) 
            #cv2.imshow("Candidate Regions", cv2_image)
            cv2.waitKey(1) 
            input()
    except:
        continue

input()

