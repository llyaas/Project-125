#C125 Model view Controller Prediction Model


#Importing all the libraries, pil = python image & sklearn library is for classification, clustering & regression algorithms etc
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

#We recieve data from given data sets & train & test the model
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#We split the data to train (7500) & test the model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
#Then we scale the data to make it of equal size
#Then we fit the data into logistic regression model, this means we need to scale the data to make sure the data points in
#X & Y are equal so we will divide them using 225 which is the maximum quick 
#After scaling the data we fit it inside our model so that it can give an output with maximum accuracy
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

#Each frame acts as an image itself where we do some proccessing & then predict the value from it
#We create a function which will take the image as a parameter & convert it to scaler quantity
#And make it grey so the colours don't affect the prediction
def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    #Resize the frame into 28x28 scales
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    #Using the percentile function we get the minimum pixel
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    #Using cliff function each image is given a number
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    #Based on that we get the maximum pixel & make it into an array or join all the values into an array
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    #Creating a test sample & making predicitons based on this sample
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]