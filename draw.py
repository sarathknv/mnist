"""Classify hand written digits.
    (real-time)

    - Run the script.
    - draw the digit in the popped up window.

# Controls:
    'space' -  reset
    'esc'   -  close

'''
ToDo: add another class, which denotes not a number.
        Here, even a black image is classified as '1' with 0.9 prob.
'''
"""
import sys
import numpy as np
import cv2
from keras.models import load_model
from maxmin import MaxMin

model = load_model('saved_models/best_val_acc_epoch_20_bs_4.h5', custom_objects={'MaxMin': MaxMin})

window = 'Draw'
cv2.namedWindow(window)

scale = 10
img = np.zeros((28*scale, 28*scale), dtype=np.uint8)
img2 = img.copy() # for reset
pt_prev = None


def draw(event, x, y, flags, param):
    global pt_prev
    pt = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        pt_prev = pt
    elif event == cv2.EVENT_LBUTTONUP:
        pt_prev = None
    # flags 33 for LBUTTON kept pressed cv2.EVENT_FLAG_LBUTTON is 1 always
    # flags & cv2.EVENT_FLAG_LBUTTON == 1 #### cv2.EVENT_FLAG_RBUTTON is 3
    if pt_prev and flags & cv2.EVENT_FLAG_LBUTTON:
        cv2.line(img, pt_prev, pt, (255), 5)
        pt_prev = pt

cv2.setMouseCallback(window, draw)

while True:
    img_28 = cv2.resize(img, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_AREA)
    # when scaled down max pixel value doesn't remain 255
    img_28[img_28 != 0] = 255
    img_28 = img_28.astype(np.float32) / 255.0
    # input img_28 to the trained model --- [batch_size, rows, cols, channels]
    pred = model.predict(img_28[None, :, :, None])[0] # list of list....batch 1
    prob = [(i, prob) for i, prob in enumerate(pred)]
    # prob now is a list of tuples, each tuple -- (number, probability)
    prob.sort(key=lambda p:p[1], reverse=True)

    #cv2.putText(img, str(prob[0][0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 1)
    print 'predicted number: ', prob[0][0], 'probability: ', prob[0][1], '\r',
    sys.stdout.flush() # when printing in the same line output is not flushed regularly, so we do it

    cv2.imshow(window, img)

    key = cv2.waitKey(100) & 0xFF
    if key == 27:
        break
    elif key == 32:
        img = img2.copy()
cv2.destroyAllWindows()
