#  HANDWRITTEN DIGIT RECOGNITION PROJECT
# We will use a well known dataset named MNIST 784.
# The dataset contains gray-scale images of hand-drawn digits, from zero through nine.
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.
# In training set there are 60,000 images and in test set we have 10,000 images for testing purpuse.

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml  # for downloading mnist dataset...
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# This operation can take 1-2 minutes..
mnist = fetch_openml('mnist_784',parser='auto')
print(mnist.data.shape,mnist.feature_names)
# Let's define a function to see the number images in the Mnist dataset:
# Let it take the dataframe and index number of the dataframe as arguments.
def showimage(dframe, index):
    some_digit = dframe.to_numpy()[index]
    print(some_digit)
    some_digit_image = some_digit.reshape(28,28)
    print(some_digit_image)
    plt.imshow(some_digit_image,cmap="binary")
    plt.axis("off")
    plt.show()
# Sample usage:
showimage(mnist.data, 0)
# 0,000 image files, 784 sizes (784 features) for each image.
# Split Data -> Training Set and Test Set
# test and train ratio ise 1/7 and 6/7
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)
print('Train_img type:',type(train_img))
# We are copying train_img for later use (for checking results)..
test_img_copy = test_img.copy()
showimage(test_img_copy, 2)
# We need to scale our data:
# Because PCA can give erroneous results in unscaled data, so we definitely go through the scaling process.
# For this purpose, we use StandardScaler...
scaler = StandardScaler()
# We only need to make the scaler fit on the training set.
scaler.fit(train_img)
# But we need to transform both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)
# Stage 1: We will apply PCA Algorithms on the dataset..
# We state that we want Variance to be protected by 95%...
# # Make an instance of the PCA Model
pca_model = PCA(.95)
# We only need to do PCA to the training set: (it may take 1 minute)
pca_model.fit(train_img)
# Let's see how many dimensions it can reduce from 784 dimensions -> ? (preserving 95% variance of course)
print(pca_model.n_components_)
# Now let's reduce the size of both our train and test datasets from 784 to 327 with the PCA transform operation:
train_img = pca_model.transform(train_img)
test_img = pca_model.transform(test_img)
# Stage 2: Logistic Regression
# Now we will apply our Logistic Regression model, which is our 2nd Machine Learning model, on our PCA-processed data.
# Since the default solver works very slowly, we create our logisticregression object by choosing the faster 'lbfgs' solver.
logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter=10000)
# We train our LogisticRegression Model using our train data:
# (it can take a few minutes)
logisticRegr.fit(train_img, train_lbl)
# Our model has been trained, now let's perform machine learning recognition operation on handwritten digits:
logisticRegr.predict(test_img[0].reshape(1,-1))
showimage(test_img_copy, 0)
logisticRegr.predict(test_img[1].reshape(1,-1))
showimage(test_img_copy, 1)
showimage(test_img_copy, 42)
logisticRegr.predict(test_img[42].reshape(1,-1))
showimage(test_img_copy, 57)
logisticRegr.predict(test_img[57].reshape(1,-1))
# Measuring the accuracy of our model
# We use score method for this purpose:
print(logisticRegr.score(test_img, test_lbl))
# The Conclusion
# In this project, we significantly shortened the training time of AI by logistic regression using PCA. I aimed to protect 95% variance. You can find out for yourself how much the time is shortened by lowering the 95% variance. The PCA algorithm, which greatly shortens the training time of artificial intelligence for 10 digits, will shorten the training time of artificial intelligence for hundreds or even thousands of different object types by hours, and this will make your programs run much faster.
# With this project, we brought together 2 completely different machine learning models and accomplished a task that we can encounter in daily life by writing an artificial intelligence program on our computer!
# It's a super technology when we really think about it, we can automatically recognize handwritten numbers and convert it into a digital number character that the computer can understand! If you think about it, you can come up with a wide variety of inventions and products using such technology. For example... you could develop this project a little further (including the letters in the alphabet) and create a software that automatically reads handwritten petitions and converts them to a digital Microsoft Word file or PDF file. People can automatically upload a petition that they create by taking their photo, and you can save it as text in the database. Thanks to artificial intelligence, you will have skipped many manual steps in between!
