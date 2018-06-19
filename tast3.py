import time

import matplotlib.pyplot as plt
import random
from sklearn import datasets
from sklearn import svm
from sklearn import neighbors
from sklearn import naive_bayes


#Load the built-in dataset that comes with the sklearn module
digits = datasets.load_digits()

#Define ML classifier algorithms we are going to test out
classifier = svm.SVC(kernel="linear")  #support vector machine()
#classifier = neighbors.KNeighborsClassifier(10)  #K Nearest-Neighbors
#classifier = naive_bayes.GaussianNB()  #Naive Bayes


#Total number of digit images in dataset
num_images = len(digits.data)
print "Total number of digit images in dataset: ",num_images

#For a classification task, we have to define a training and test set
#Define which images to use for training the model
#We will then test the prediction on images outside of this
#General rule of thumb, train on 60% of the data, and test on the remaining 40%
num_training=1080

#Provide information on the image data, and the true labels of the data
x,y = digits.data[1:num_training], digits.target[1:num_training]
classifier.fit(x,y)

#Here, we will test the model prediction on 10 randomly selected images
#Note that we are only choosing from the test data (that the model didn't train on)
user_correct = 0
machine_correct = 0
print "Time to guess!"
print
print

for i in random.sample(range(num_training,num_images), 10):
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()

    # Prompt user for input
    human_prediction = input("Type the number your saw: ")
    human_prediction = int(human_prediction)

    # scikitlearn wants us to reshape 1D araays like this
    number_label = digits.target[i].reshape(1,-1)[0][0]
    print 'The correct answer is: ', number_label

    # Predict the number using your chosen classifier
    machine_prediction = classifier.predict( digits.data[i].reshape(1,-1) )[0]
    print 'Our chosen model predicted: ', machine_prediction, "!!!"
    print ""

    #Keep track of score for humans vs machines
    if human_prediction == number_label:
        user_correct+=1
    if machine_prediction == number_label:
        machine_correct+=1

    time.sleep(1)


########################################
# Print results of the competition.
print "Finished!"
print "The human identified %i out of 10 numbers" % user_correct
print "The machine correctly identified %i out of 10 numbers" % machine_correct

# Message about humans vs machines.
if user_correct < machine_correct:
    print("Machines > humans!")
elif user_correct == machine_correct:
    print("Humans and machines coexist")
else:
    print("Humans > machines!")

#Save the results
my_results="%d,%d\n" % (user_correct,machine_correct)
with open("data/human_vs_machine.csv", "a") as myfile:
    myfile.write(my_results)
