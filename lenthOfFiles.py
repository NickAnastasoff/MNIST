import os
dog_test = len(os.listdir("test/dogs"))
print("Number of dog test images: {}".format(dog_test))

cat_test = len(os.listdir("test/cats"))
print("Number of cat test images: {}".format(cat_test))

dog_train = len(os.listdir("train/dogs"))
print("Number of dog train images: {}".format(dog_train))

cat_train = len(os.listdir("train/cats"))
print("Number of cat train images: {}".format(cat_train))