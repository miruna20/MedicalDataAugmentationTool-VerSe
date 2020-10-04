import tensorflow as tf
import numpy as np
import nibabel as nib

#load an estimation and an image and calculate the l2 distance in 3 ways

#load prediction
prediction = nib.load("/home/payer/training/debug_train/succ_examples/estimation_verse005_23_500.nii.gz")
y_pred = np.array(prediction.dataobj)

#load shuffled image as GT just for the test
GT = nib.load("/home/payer/training/debug_train/succ_examples/image_verse005_23_500.nii.gz")
y_true = np.array(GT.dataobj)
#y_true = [[0., 1.], [0., 0.]]
#y_pred = [[1., 1.], [1., 0.]]

mse = tf.keras.losses.MeanSquaredError()
mse_keras = mse(y_true, y_pred)
mse_tf = tf.losses.mean_squared_error(y_true,y_pred)

mse_manuallycalc =  np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)

with tf.Session() as sess:
    print("with keras:")
    print(mse_keras.eval())
    print("with tf:")
    print(mse_tf.eval())
    print("manually calculated:")
    print(mse_manuallycalc)

