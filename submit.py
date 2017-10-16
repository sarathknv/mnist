import numpy as np
import pandas as pd
from keras.models import load_model
#from keras_maxmin_impl import MaxMinConvolution2D
from maxmin import MaxMin

model = load_model('saved_models/best_val_acc_epoch_20_bs_4.h5', custom_objects={'MaxMin': MaxMin})
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print 'model loaded'

test = pd.read_csv('dataset/test.csv')

test = test.values.reshape(-1, 28, 28, 1)
test = test/255.0
print 'test data loaded.'
results = model.predict(test)

results=np.argmax(results, axis=1)
results=pd.Series(results, name='Label')
submission=pd.concat([pd.Series(range(1,28001), name='ImageId'), results], axis=1)
submission.to_csv('new_epoch_20_bs_4.csv', index=False)
