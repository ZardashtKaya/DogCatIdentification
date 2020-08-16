import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pickle
import random
import cv2
from tqdm import tqdm
import sys
IMG_SIZE = 50
IMG_SIZE = int(sys.argv[2])
# testing_image = "./test.jpg"

testing_image = sys.argv[1]


img_array = cv2.imread(testing_image, cv2.IMREAD_GRAYSCALE)  # convert to array
# resize to normalize data size
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# add this to our testing_data
# testing_data.append([new_array, class_num])
testing_data = new_array


print(testing_data)

plt.imshow(testing_data, cmap='gray')  # graph it
plt.show()  # display!
testing_data = np.array(testing_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(testing_data)
print("Loading Model...")
# model_name ='dogCatIden'+str(IMG_SIZE)+'.model'
model_name ='dogCatIden.model'
new_model = tf.keras.models.load_model(model_name)


testing_data = testing_data/255.0
IMG_SIZE = 50
print("Predicting...")
prediction = new_model.predict(testing_data)

if(prediction > 0.5):
    print("It's a Cat")
else:
    print("It's a Dog")
