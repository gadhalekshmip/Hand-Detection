import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
import os

L = os.listdir("D:\\foldername\")
print(L)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

thshld = 50

hand_det = mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]



c=0   #filename starting
i=0
while True:

    ret,frame = cam.read()
    #print(1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_img = np.zeros(img.shape)
    results = hand_det.process(img)
    # img = blank_img
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                    blank_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            x,y,x2,y2 = calc_bounding_rect(img, hand_landmarks)
            cv2.rectangle(img, (x-thshld, y-thshld), (x2+thshld, y2+thshld), (0, 255, 0), 3)
            # im = Image.fromarray(blank_img)
            # im = im.crop((x,y,x2,y2))
            # im = np.array(im)q
            im = blank_img
            im = cv2.resize(im, (128,128))
          
            # det.forward(im)
            


    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        

    # if cv2.waitKey(1) == ord('q'):
    #     break

    if cv2.waitKey(1) == ord('s'):
        print(f"Saved to {L[i]}")
        cv2.imwrite(f"D:\\foldername\\{L[i]}\\{c}.jpg", blank_img)
        c+=1
        if c > 20:# till c=20
            c=0
            i+=1
            print(f"Now show {L[i]}")
    
    cv2.rectangle(img, (0,0), (200,200), (0,0,255), 4)
    cv2.imshow('img',blank_img)
    # cv2.imshow("image",img)
    # time.sleep(1)



cam.release()
cv2.destroyAllWindows()
