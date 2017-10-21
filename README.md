# MNIST Digit Classifier Real-Time

This is a 4 layer Convolutional Neural Network written in Keras. [MaxMin CNN](https://github.com/karandesai-96/maxmin-cnn) layers were used instead of the traditional CNN layers.  


## Overview of executable scripts:
`draw.py`   &nbsp;&nbsp;&nbsp;classify hand drawn digits using mouse in real-time.  
`maxmin.py` implementation of [MaxMin CNN](https://github.com/karandesai-96/maxmin-cnn).  
`submit.py` for Kaggle [competition](https://www.kaggle.com/c/digit-recognizer).Generates predictions on test data.  
`train.py`  &nbsp;&nbsp;train the network.  


## Training:

Run the script `python train.py --epochs 20 --batch_size 4`

Proper training can give 0.996 accuracy on the test data in [Kaggle Digit Recognizer Challenge](https://www.kaggle.com/c/digit-recognizer).
The model `best_val_acc_epoch_20_bs_4.h5` gives a score of 0.99171 on test data. It was trained for 20 epochs with batch size of 4. 
       ![Accuracy and Loss plot](https://github.com/sarathknv/mnist/blob/master/pics/epochs_20_bs_4.png)
       
       
       
## Testing the classifier Real-Time:
 Run the script `python draw.py` &nbsp;Control keys: `space` reset and `esc` close   
 
 ![Sccreenshot draw.py](https://github.com/sarathknv/mnist/blob/master/pics/one.png)

## Requirements:
* Python 2
* Keras 2
* sklearn 0.19
* OpenCV 2
* pandas, numpy, matplotlib, argparse
