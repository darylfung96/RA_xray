import cv2
import matplotlib.pyplot as plt

joint_cascade = cv2.CascadeClassifier('data/cascade_joint.xml')


def detect_joint(img):
    joint_img = img.copy()

    joint_rect = joint_cascade.detectMultiScale(joint_img, scaleFactor=1.1, minNeighbors=15)

    for (x, y, w, h) in joint_rect:
        cv2.rectangle(joint_img, (x, y), (x+w, y+h), (255, 255, 255), 10)
    return joint_img


img = cv2.imread('train_img/4.jpg')
joint_img = detect_joint(img)
plt.imshow(joint_img)
plt.show()