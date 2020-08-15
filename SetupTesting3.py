import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
from tqdm import tqdm
DATADIR = "./Data/Testing"
IMG_SIZE = 50
IMG_SIZE = sys.argv[1]
CATEGORIES = ["dogs", "cats"]

testing_data = []


def create_testing_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        # get the classification  (0 or a 1). 0=dog 1=cat
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                # resize to normalize data size
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # add this to our testing_data
                testing_data.append([new_array, class_num])
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            # except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            # except Exception as e:
            #    print("general exception", e, os.path.join(path,img))


create_testing_data()

# print(len(testing_data))
print(testing_data)


X = []
y = []

for features, label in testing_data:
    X.append(features)
    y.append(label)

print(y)
print(X)
# print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(X)
print(y)

pickle_out = open("X_test.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
