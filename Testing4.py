import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
pickle_in = open("X_test.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y_test.pickle", "rb")
y = pickle.load(pickle_in)

new_model = tf.keras.models.load_model('dogCatIden.model')

# print(X)
# print(y)
X = X/255.0

# print(X)

val_loss, val_acc = new_model.evaluate(X, y)
# predictions = new_model.predict(X)
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy
IMG_SIZE = 50
predictions = new_model.predict([X])


def pC(pred):
    if(pred > 0.5):
        return "cat"
    else:
        return "dog"


# print(X[0])
print(pC(predictions[24]))
