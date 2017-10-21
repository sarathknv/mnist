# MNIST Digit Classifier

This is a 4 layer Convolutional Neural Network written in Keras. [MaxMin CNN](https://github.com/karandesai-96/maxmin-cnn) layers were used instead of the traditional CNN layers. 

## Overview of the executable scripts:
## Testing the classifier Real-Time:

### Training

Run the script `python train.py --epochs 20 --batch_size 4`

Proper training can give 0.996 accuracy on the test data in [Kaggle Digit Recognizer Challenge](https://www.kaggle.com/c/digit-recognizer).
The model `best_val_acc_epoch_20_bs_4.h5` gives a score of 0.99171 on test data. It was trained for 20 epochs with batch size of 4. 
       ![Accuracy and Loss plot](https://github.com/sarathknv/mnist/blob/master/pics/epochs_20_bs_4.png)
       
       
       
