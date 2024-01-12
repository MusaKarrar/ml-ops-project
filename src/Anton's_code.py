import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import math
import sys
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import csv
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error


number_of_bands = 4
# Specify the path to the folder containing your image data

# training1 test1
# finaltraining finaltest
data_traning = 'data\raw\training5'
data_test = 'data\raw\test5'


# Create empty lists to store the data and labels
X_train = np.array([], dtype=np.float16)
X_test = np.array([], dtype=np.float16)

all_type_img_train = np.array([[]], dtype=np.float16)
all_type_img_test = np.array([[]], dtype=np.float16)
imageCatagory = 0
y_train = np.array([], dtype=np.float16)
y_test = np.array([], dtype=np.float16)

# Loop through the image files in the folder
for filename in os.listdir(data_traning):

    imageCatagory = imageCatagory+1
    if filename.endswith('.tif'):  # Adjust the file format to match your data
        # Load the image using OpenCV
        all_type_img_train = np.array([[]], dtype=np.float16)

        img = cv2.imread(os.path.join(data_traning, filename),
                         cv2.IMREAD_UNCHANGED)

        if img is not None:
            all_type_img_train = np.append(
                all_type_img_train, img.flatten())

            del img
        else:
            print("train", filename)
        #   img.flatten()
        # cv2.mean(filtered_pixels)

        # Extract the label from the filename or use a different method to assign labels
        # For example, you can name your files like "class_label_image1.tif"
        # Extract the class label from the filename¨
    X_train = np.append(X_train, (all_type_img_train ))
    del all_type_img_train

    if imageCatagory == number_of_bands:
        #        print("the last 4 images have been catagolize")
        imageCatagory = 0
        if 'zero' in filename:
            y_train = np.append(y_train, 0)
        elif '100kg' in filename:
            y_train = np.append(y_train, 100)
        elif '200kg' in filename:
            y_train = np.append(y_train, 200)
        elif '300kg' in filename:
            y_train = np.append(y_train, 300)
#    print("time it tooke to load imamge: ", elapsed_time)


# print("the training set have been loaded into an array")
for filename in os.listdir(data_test):
    all_type_img_test = np.array([[]], dtype=np.float16)

    start_time = 0
    end_time = 0
    start_time = time.time()
    imageCatagory = imageCatagory+1

    if filename.endswith('.tif'):  # Adjust the file format to match your data
        # Load the image using OpenCV

        img = cv2.imread(os.path.join(data_test, filename),
                         cv2.IMREAD_UNCHANGED)
        if img is not None:

            all_type_img_test = np.append(
                all_type_img_test, img.flatten())
#            print(f"picture {filename} has been loaded")
            del img

        else:
            print("test", filename)
        # img.flatten()
        # cv2.mean(filtered_pixels)

        # Extract the label from the filename or use a different method to assign labels
        # For example, you can name your files like "class_label_image1.tif"
        # Extract the class label from the filename
    X_test = np.append(X_test, (all_type_img_test ))
    del all_type_img_test

    if imageCatagory == number_of_bands:
        imageCatagory = 0
#        print("the last 4 images have been catagolize")
        if 'zero' in filename:
            y_test = np.append(y_test, 0)
        elif '100kg' in filename:
            y_test = np.append(y_test, 100)
        elif '200kg' in filename:
            y_test = np.append(y_test, 200)
        elif '300kg' in filename:
            y_test = np.append(y_test, 300)
    end_time = time.time()
#    print("time it tooke to load imamge: ", elapsed_time)
# print("the test set have been loaded into an array")

# Create and train a Random Forest classifier
model = SVR(kernel='linear', C=1e3, gamma=0.1)


print("traning the model")
print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))

X_train = X_train.reshape(len(y_train), 16960*number_of_bands)
X_test = X_test.reshape(len(y_test), 16960*number_of_bands)
n_components = 120  # Example value, adjust it based on your needs

# Initialize PCA
#pca = PCA(n_components=n_components)

# Fit PCA on the training data and transform both training and test data
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
print("y data: ", y_train.shape)
print("x data: ", X_train.shape)
print("y data: ", y_train)
print("x data: ", X_train)

model.fit(X_train, y_train)

print("The model have been trained")

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

plt.scatter(y_pred, y_test, color="black", s=30)

# x-axis label
plt.xlabel('Predicted Nitrogen (kg)')
# frequency label
plt.ylabel('Nitrogen (kg)')
# plot title
plt.title('SVR Prediction of Nitrogen using PCA om 100 component')
# showing legend
plt.legend()

# function to show the plot
#we have to change this path
plt.savefig(r"C:\Users\Anton\OneDrive - Danmarks Tekniske Universitet\Skrivebord\Arial tools\Report\Plots\svr_scatter.png")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)


print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("RMSE:", math.sqrt(mse))
r2_score(y_test, y_pred)
print("r^2", r2_score)
print(y_pred)
print(y_test)
dataframe = pd.DataFrame(y_pred)
dataframe.to_csv(
    r"C:\Users\Anton\OneDrive - Danmarks Tekniske Universitet\Skrivebord\Arial tools\Code\svr.csv")
r, p = stats.pearsonr(y_pred, y_test)
r2 = r**2
print('r^2', r2)
