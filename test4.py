#!/bin/python

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

def match_rate(test_data,model):
  #Initialize count for matches (predicted and true labels)
  count = 0.0
  for s in test_data:
    #Check if the true and predicted label are the same
    if s[1] == model.classify(s):
      count+=1
  num_correct = count/len(test_data)
  return(round((num_correct*100),2))

#Remember, for both training and test sets, we always tell the classifiers what the TRUE labels are, before we can predict something new and unknown
#This is how all supervised learning works
train = [
      ('I love this sandwich.', 'positive'),
      ('this is an amazing place!', 'positive'),
      ('She feels very good about these ideas.', 'positive'),
      ('this is my best work.', 'positive'),
      ("what an awesome view", 'positive'),
      ('We do not like this restaurant', 'negative'),
      ("I'm tired of this stuff.", 'negative'),
      ("I can't deal with this", 'negative'),
      ('he is my sworn enemy!', 'negative'),
      ('my boss is horrible.', 'negative')
  ]
test = [
      ('the beverage was good.', 'positive'),
      ('I do not enjoy my job', 'negative'),
      ("he ain't feeling dandy today.", 'negative'),
      ("I feel amazing!", 'positive'),
      ('Gary is a friend of mine.', 'positive'),
      ("I can't believe I'm doing this.", 'negative'),
      ("she ain't from around here",'negative'),
      ("that sandwich made me sick",'negative')
  ]
 
#Train the classifier using NB algorithm on the training sentences provided
cl = NaiveBayesClassifier(train)
 
#The classify() function will test your classifier on a given test sentence
print "Applying Naive Bayes classifer to test sentences"
print

for s in test:
  print "Sentence: ",s[0]
  print "Predicted Connotation: ",cl.classify(s)
  print "True Connotation: ",s[1]
  print
#Based on the given training and test data, check the accuracy of this classifier
print "Current accuracy of model: ",match_rate(test,cl), " %"
print
 
#We can print out the top most informative features that the classifier made use of, given the training data
print "Top 5 most informative features according to NB classifier:"
cl.show_informative_features(5)
print
 
 #Update the classifier with new data (it will re-train including these new sentences)
new_data = [('she is an amazing friend.', 'positive'),
             ("I'm happy to have a new friend.", 'positive'),
             ("machine learning is awesome!",'positive'),
             ("I'm feeling dandy today.", 'negative'),
             ("Gary wishes he didn't have a meeting",'negative'),
             ("he ain't from around here.", 'negative'),
             ("I do not understand", 'negative')]
cl.update(new_data)

#See if the accuracy improved by testing it on the test dataset
print "New accuracy of updated model: ",match_rate(test,cl), " %"
print

#Now the fun part
#Give this classifier a totally new sentence of your own (that it hasn't seen before) to see if it gets it right!
new_sentence = TextBlob("I dislike this sandwich. I won't feel so great later",classifier=cl)
print "New sentence: ", new_sentence
print "Predicted Connotation: ", new_sentence.classify()
print

#Let us split this sentence into it's components and test the classifier on each sentence individually
#See if there is any difference in its prediction
print "Looking at the prediction of each part of the sentence separately.."
print

for s in new_sentence.sentences:
 print "Sentence: ", s
 print "Predicted Connotation: ", s.classify()
 print 
