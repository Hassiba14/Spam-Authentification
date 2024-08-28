import pandas as pd
emails = pd.read_csv('Activity_20.2.csv')

In this activity, you will apply the Naïve Bayes theorem to a database that contains spam emails.
Naïve Bayes is a classification technique based on Bayes theorem with an assumption of independence among predictors. 

Bayes theorem is stated mathematically as the following equation:

$$P(A | B) = \frac{P(B|A)P(A)}{P(B)},$$

where

 - $P(A| B)$ is a conditional probability: the likelihood of event $A$ occurring given that $B$ is true.
 - $P(B|A)$ is also a conditional probability: the likelihood of event $B$ occurring given that $A$ is true.
 - $P(A)$ and $P(B)$ are the probabilities of observing events $A$ and $B$.

In simple terms, a Naïve Bayes classifier assumes that the presence of a particular feature in a *class* is unrelated to the presence of any other feature. 

For example, a fruit may be considered to be an apple if it is red, round, and about three inches in diameter. Even if these features depend on each other or upon the existence of the other features, these properties independently contribute to the probability that this fruit is an apple. That is why it is known as ‘Naïve’.

This activity is designed to help you apply the machine learning algorithms that you have learned using *packages* in Python. Python concepts, instructions, and the starter code are embedded within this Jupyter Notebook to help guide you as you progress through the activity. Remember to run the code of each code cell prior to your submission. Upon completing the activity, you are encouraged to compare your work against the solution file to perform a self-assessment.

## Index:

- [Part 1 - Importing the Dataset and Exploratory Data Analysis (EDA)](#part1)
- [Part 2 - Shuffling  and Splitting the Emails](#part2)
- [Part 3 - Building a Simple Naïve Bayes Classifier From Scratch](#part3)
- [Part 4 - Explaining the Code Given in Part 3](#part4)
- [Part 5 - Train the Classifier `train`](#part5)
- [Part 6 - Explore the Performance of `train` Classifier ](#part6)
- [Part 7 - Training the `train2` Classifier ](#part7)


## The Naïve Bayes Algorithm

You have now learned about the **Naïve Bayes algorithm** for classification. 

The pseudo-algorithm for Naïve Bayes can be summarized as follows: 
1. Load the training and test data.
2. Shuffle the messages and split them.
3. Build a simple Naïve Bayes classifier from scratch.
4. Train the classifier and explore the performance.
###  Build a Naïve Bayes Spam Filter

For this exercise, you will use the [“SMSSpamCollection”](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/) dataset  to build a Naïve Bayes spam filter by going through the following steps:

1. Load the data file.
2. Shuffle the messages and split them into a training set (2,500 emails), a validation set (1,000 messages), and a test set (all remaining messages).
3. Build a simple Naïve Bayes classifier.
4. Explain the code given in Part 3.
5. Use your training set to train the ‘train’ classifier. Note that the interfaces of the classifiers require you to pass the non-spam and spam emails separately.
6. Using the validation set, explore how the classifier performs.
7. Define a second classifier and compare its performance with the one defined in Part 3.

[Back to top](#Index:) 

<a id='part1'></a>

### Part 1: Importing the Dataset and  Exploratory Data Analysis (EDA)

Begin by using `pandas` to import the dataset. To do so, import `pandas` first and then read the file using the `.read_csv` *function* by passing the name of the dataset that you want to read as a *string*.

import pandas as pd
pd.read_csv('A tivity_20.2.csv')
    
Notice that because the rows in the dataset are separated using a `\t`, the type of delimiter is specified in the `.read_csv()` *function*; the default value is `,`. Additionally, the list of column names to use is specified (`"label"` and `"emails"`).

import pandas as pd

emails = pd.read_csv('Activity_20.2.csv', sep = '\t', names = ["label", "email"])
    
Before performing any algorithm on the *dataframe*, it is always good practice to perform exploratory data analysis.

Begin by visualizing the first ten rows of the `df` *dataframe* using the `head()` *function*. By default, *`head()`* displays the first five rows of a *dataframe*.

Complete the code cell below by passing the desired number of rows to the `head()` *function* as an *integer*.

head(10)

Next, you can retrieve some more information about your *dataframe* by using the `shape` and `columns` *methods* and the `describe()` *function*.

Here's a brief description of what each of the above *functions* does:
- `shape`: returns a *tuple* representing the dimensionality of the *dataframe*.
- `columns`: returns the column labels of the *dataframe*.
- `describe()`: returns summary statistics of the columns in the *dataframe* provided, such as mean, count, standard deviation, etc.

df.shape()
df.ddescribe()
    
Run the cells below to review the information about the *dataframe*.
emails.shape
emails.columns
emails.describe()
[Back to top](#Index:) 

<a id='part2'></a>

### Part 2: Shuffling  and Splitting the Text Messages

In the second part of this activity, you will shuffle the emails and split them into a training set (2,500 messages), a validation set (1,000 messages), and a test set (all remaining messages).

Begin by shuffling the messages. This can be done using the `sample` *function* from the `pandas` *library*.

Complete the code cell below by applying the `sample()` *function* to the *dataframe* emails. Set the `frac = 1` *argument* and `random_state = 0`. `frac` denotes the proportion of the *dataframe* to sample, and `random_state` is a random seed that ensures reproducibility of your results. 

emails = random.shuffle(emails)
    
Next, reset the *index* of `emails` to align with the shuffled emails by using the `reset_index` *function* with the appropriate *argument*. 

samples_df = df. sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    
Here's a link for you where you can find the documentation about the [`reset_index()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html) *function*.
emails = head()

Run the code cell below to visualize the updated *dataframe*.
emails.head()
Next, you need to split the emails into a training set (the first 2,500 emails in the *dataframe*), a validation set (the next 1,000 emails), and a testing set (the remaining emails). These three sets will be defined as `trainingMsgs`, `valMsgs`, and `testingMsgs`.

In the code cell below, the messages and their correspoding labels are defined. Next, you will split the messages into the required sets according to the instructions.
email = list(emails.email) 
lbls =list(emails.label) 
trainingEmail = email[:2500] 
valEmail = email[2500:3500] 
testingEmail = email[3500:]
Following the syntax used above, complete the cell below to split the labels into a training set, a validation set, and a test set.
traininglbls =
vallbls = 
testinglbls = 
[Back to top](#Index:) 

<a id='part3'></a>

### Part 3: Building a Simple Naïve Bayes Classifier From Scratch

While Python’s SciKit-learn *library* has a [Naïve Bayes classifier](https://scikit-learn.org/stable/modules/naive_bayes.html), it works with continuous probability distributions and assumes numerical features. 

import numpy as np
    
import sklearn 


class NaiveBayesForSpam:
    def train (self, nonSPamMessages, spamMessages):
        self.words = set (' '.join (nonSPamMessages + spamMessages).split())
        self.priors = np.zeros (2)
        self.priors[0] = float (len (nonSPamMessages)) / (len (nonSPamMessages) + len (spamMessages))
        self.priors[1] = 1.0 - self.priors[0]
        self.likelihoods = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in nonSPamMessages if w in m])) / len (nonSPamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len (spamMessages)
            self.likelihoods.append ([min (prob1, 0.95), min (prob2, 0.95)])
        self.likelihoods = np.array (self.likelihoods).T
        
    def predict (self, message):
        posteriors = np.copy (self.priors)
        for i, w in enumerate (self.words):
            if w in message.lower():  # convert to lower-case
                posteriors *= self.likelihoods[:,i]
            else:                                   
                posteriors *= np.ones (2) - self.likelihoods[:,i]
            posteriors = posteriors / np.linalg.norm (posteriors)  # normalize
        if posteriors[0] > 0.5:
            return ['ham', posteriors[0]]
        return ['spam', posteriors[1]]    

    def score (self, messages, labels):
        confusion = np.zeros(4).reshape (2,2)
        for m, l in zip (messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham':
                confusion[0,0] += 1
            elif self.predict(m)[0] == 'ham' and l == 'spam':
                confusion[0,1] += 1
            elif self.predict(m)[0] == 'spam' and l == 'ham':
                confusion[1,0] += 1
            elif self.predict(m)[0] == 'spam' and l == 'spam':
                confusion[1,1] += 1
        return (confusion[0,0] + confusion[1,1]) / float (confusion.sum()), confusion
[Back to top](#Index:) 

<a id='part4'></a>

### Part 4: Explaining the Code Provided in Part 3

Before explaining the code that was provided in Part 3, it is important to have some level of intuition as to what a spam email message might look like. Usually they have words designed to catch your eye and, in some sense, to tempt you to open them. Also, spam emails tend to have words written in all capital letters and use a lot of exclamation marks.

The `train` *function* calculates and stores the prior probabilities and likelihoods based on the training dataset. In Naïve Bayes, this is all the training phase does. 

The `predict` *function* repeatedly applies the Naïve Bayes theorem to every word in the constructed *dictionary* and, based on the posterior probability, it classifies each email as `spam` or `non-spam`. The `score` *function* calls `predict` for multiple emails and compares the outcomes with the supplied `ground truth` labels, thus evaluating the classifier. It also computes and returns a confusion matrix.

[Back to top](#Index:) 

<a id='part5'></a>

### Part 5: Training the `train`  Classifier

Looking at the definition of the `train` *function* in Part 2, you can see that the training *functions* require the `non-spam` and `spam` emails to be passed on separately.

spamEmails = [m for(m,1) in zip(trainingEmail, traininglbls) if 'spam' in l]

    
    he `non-spam` emails can be passed using the code given below:
nonSpamEmails = [m for (m, l) in zip(trainingEmail, traininglbls) if 'ham' in l]

Complete the cell below to pass the spam emails.
spamEmails = 

Run the cell below to see how many `non-spam` and `spam` emails you have.
    
print(len(nonSpamEmails))
print(len(spamEmails))
Their sum equals 2,500 as expected. 

Next, you need to create the classifier for your analysis using the `NaiveBayesForSpam`*function*. 

clf = NaiveBayesForSpam()

After that, you will train `non-spammsgs` and `spammsgs` using the `train` *function*. 
    
clf = NaiveBayesForSpam()
clf.train(nonSPamEmails, spamEmails)

[Back to top](#Index:) 

<a id='part6'></a>

### Part 6: Exploring the Performance of the `train` Classifier

You can explore the performance of the two classifiers on the *validation set* by using the `.score()` *function*.



Complete the code cell below to compute the score and the confusion matrix for this case.
score. confusion = clf.score(valEmail, vallbls)

**IMPORTANT NOTE: Results in the following sections will change. This is expected and due to the random shuffling. The results will be different for each shuffling. To ensure reproducible results, define `random_state` in the `sample` *method* when shuffling the data in [Part 2: Shuffling and Splitting the Text Messages](#part2).**

score, confusion = clf.score (valEmail, vallbls)

Run the code cells below to print the score and the confusion matrix.
print("The overall performance is:", score)
print("The confusion matrix is:\n", confusion)
Your data is not equally divided into the two *classes*. As a baseline, let’s see what the success rate would be if you always guessed non-spam.

Run the code cell below to *print* the new score.
print('new_score', len([1 for l in vallbls if 'ham' in l]) / float (len ( vallbls)))
Compare the baseline score to the performance on the validation set. Which is better?

You can also calculate the sample error by calculating the score and the confusion matrix on the training set.
    
#Note: this cell may take a LONG time to run!
score, confusion = clf.score (trainingEmail, traininglbls)

score_training, confusion_training = clf.score(training€mail, traininglbls)

    

print("The overall performance is:", score)
print("The confusion matrix is:\n", confusion)
[Back to top](#Index:) 

<a id='part7'></a>

### Part 7: Training the `train2` Classifier

In this section, you will define a second classifier, `train2`, and compare its performance to the `train` classifier defined above.

The `train2` classifier is defined in the code cell below.
class NaiveBayesForSpam:
    def train2 ( self , nonSpamMessages , spamMessages) :
            self.words = set (' '.join (nonSpamMessages + spamMessages).split())
            self.priors = np. zeros (2)
            self.priors [0] = float (len (nonSpamMessages)) / (len (nonSpamMessages) +len( spamMessages ) )
            self.priors [1] = 1.0 - self . priors [0] 
            self.likelihoods = []
            spamkeywords = [ ]
            for i, w in enumerate (self.words):
                prob1 = (1.0 + len ([m for m in nonSpamMessages if w in m])) /len ( nonSpamMessages )
                prob2 = (1.0 + len ([m for m in spamMessages if w in m])) /len ( spamMessages ) 
                if prob1 * 20 < prob2:
                    self.likelihoods.append([min (prob1 , 0.95) , min (prob2 , 0.95) ])
                    spamkeywords . append (w) 
            self.words = spamkeywords
            self.likelihoods = np.array (self.likelihoods).T 
            
    def predict (self, message):
        posteriors = np.copy (self.priors)
        for i, w in enumerate (self.words):
            if w in message.lower():  # convert to lower-case
                posteriors *= self.likelihoods[:,i]
            else:                                   
                posteriors *= np.ones (2) - self.likelihoods[:,i]
            posteriors = posteriors / np.linalg.norm (posteriors)  # normalise
        if posteriors[0] > 0.5:
            return ['ham', posteriors[0]]
        return ['spam', posteriors[1]]    

    def score (self, messages, labels):
        confusion = np.zeros(4).reshape (2,2)
        for m, l in zip (messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham':
                confusion[0,0] += 1
            elif self.predict(m)[0] == 'ham' and l == 'spam':
                confusion[0,1] += 1
            elif self.predict(m)[0] == 'spam' and l == 'ham':
                confusion[1,0] += 1
            elif self.predict(m)[0] == 'spam' and l == 'spam':
                confusion[1,1] += 1
        return (confusion[0,0] + confusion[1,1]) / float (confusion.sum()), confusion
Next, you need to update the classifier for your analysis using the `NaiveBayesForSpam` *function*. Complete the cell below to create the `clf` classifier.

clf = NaiveBayesForSpam()
    
Next, train `non-spammsgs` and `spammsgs` using the `train2` *function*. 

clf.train(nonSpamEmails, spamEmails)

*HINT:* For this last part, look at the definition of the `train2` *function*.
clf = 
#train using the new classifier
clf.train(NaiveBayesForSpam)

Re-compute the score and the confusion matrix on the validation set using the updated classifier.
#Again, this cell may take a long time to run!
score_2, confusion_2 = clf.score(valEmails, valbls)
