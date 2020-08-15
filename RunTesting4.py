import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

pickle_in = open("X_test.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y_test.pickle", "rb")
y = pickle.load(pickle_in)
y=np.array(y)
new_model = tf.keras.models.load_model('dogCatIden.model')


X = X/255.0

# print(X)
print("Evaluating Model...")
val_loss, val_acc = new_model.evaluate(X, y)
# predictions = new_model.predict(X)
print("Validation Loss: ", val_loss)  # model's loss (error)
print("Validation Accuracy: ", val_acc)  # model's accuracy
IMG_SIZE = 50

predictions = new_model.predict([X])
def pC(pred):
    if(pred > 0.5):
        return "cat"
    else:
        return "dog"






i = 0
print("Running Predictions on all images...")


while i < len(X):
    print(pC(predictions[i]))
    i += 1
