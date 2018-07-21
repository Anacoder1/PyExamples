import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
print(iris_data[0], iris_data[79], iris_data[100])
print(iris_labels[0], iris_labels[79], iris_labels[100])

"""
Learnset created from the sets above
np.random is used to split the data randomly.
"""
np.random.seed(42)

indices = np.random.permutation(len(iris_data))
n_training_samples = 12
learnset_data= iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]
print(learnset_data[:4], learnset_labels[:4])
print(testset_data[:4], testset_labels[:4])

"""
Code below is only necessary to visualiza the data of learnset.
This data consists of 4 values per iris item, so we'll reduce the
data to 3 values by summing up the 3rd and 4th value.
This way, we're capable of depicting the data in 3D space.
"""
# %matplotlib inline (only do if coding in ipython notebook)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
colours = ("r", "b")
X = []
for iclass in range(3) :
    X.append([[], [], []])
    for i in range(len(learnset_data)) :
        if learnset_labels[i] == iclass :
            X[iclass][0].append(learnset_data[i][0])
            X[iclass][1].append(learnset_data[i][1])
            X[iclass][2].append(sum(learnset_data[i][2:]))
colours = ("r", "g", "y")
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
for iclass in range(3):
    ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2],
    c = colours[iclass])
plt.show()

"""
A distance function is required to find the similarity
between 2 instances. In our example, the Euclidean 
distance is ideal.
"""
def distance(instance1, instance2) :
    # in case if instances are lists or tuples:
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1 - instance2)
print(distance([3, 5], [1, 1]))
print(distance(learnset_data[3], learnset_data[44]))

def get_neighbors(training_set, labels, test_instance, k, 
                  distance = distance) :
                  """
                  get_neighbors calculates a list of the k nearest neighbors
                  of an instance 'test instance'.
                  The list neighbors contains 3-tuples with
                  (index, dist, label) where
                  index = index from the training_set,
                  dist = distance between the test_instance and the
                         instance training_set[index]
                  distance is a reference to a function used to calculate the
                  distances
                  """
                  distances = []
                  for index in range(len(training_set)) :
                      dist = distance(test_instance, training_set[index])
                      distances.append((training_set[index], dist, labels[index]))
                  distances.sort(key = lambda x: x[1])
                  neighbors = distances[:k]
                  return neighbors

# Testing the function with our iris samples
for i in range(5) :
    neighbors = get_neighbors(learnset_data, learnset_labels,
                              testset_data[i], 3, distance = distance)
    print(i, testset_data[i], testset_labels[i], neighbors)

"""
Vote function written below. It uses the class 'Counter' from collections
to count the quality of the classes inside an instance list.
This instance list will be neighbors. The function 'vote' returns the
most common class.
"""
from collections import Counter

def vote(neighbors) :
    class_counter = Counter()
    for neighbor in neighbors :
        class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]

# Testing 'vote' on our training samples

for i in range(n_training_samples) :
    neighbors = get_neighbors(learnset_data, learnset_labels,
                              testset_data[i], 3, distance = distance)
    print("index: ", i, ", result of vote: ", vote(neighbors),
          ", label: ", testset_labels[i],
          ", data: ", testset_data[i])

"""
All predictions correspond to labeled results, except the item with
index 8.
'vote_prob' function below is similar to 'vote' but returns the
class name and the probability for this class
"""

def vote_prob(neighbors) :
    class_counter = Counter()
    for neighbor in neighbors :
        class_counter[neighbor[2]] += 1
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    return winner, votes4winner / sum(votes)

for i in range(n_training_samples) :
    neighbors = get_neighbors(learnset_data, learnset_labels, 
                              testset_data[i], 5, distance = distance)
    print("index: ", i,
          ", vote_prob: ", vote_prob(neighbors),
          ", label: ", testset_labels[i],
          ", data: ", testset_data[i])

"""
'vote_harmonic_weights' below gives more preference (weight)
to neighbors of an instance which are closer than those which
are farther
"""
def vote_harmonic_weights(neighbors, all_results = True) :
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors) :
        class_counter[neighbors[index][2]] += 1 / (index + 1)
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results :
        total = sum(class_counter.values(), 0.0)
        for key in class_counter :
            class_counter[key] /= total
        return winner, class_counter.most_common()
    else :
        return winner, votes4winner / sum(votes)

for i in range(n_training_samples) :
    neighbors = get_neighbors(learnset_data, learnset_labels,
                              testset_data[i], 6, distance = distance)
    print("index: ", i,
          ", result of vote: ",
          vote_harmonic_weights(neighbors,
                                all_results = True))   

"""
Previous approach took only the ranking of neighbors according to their
distance in account. We can improve the voting by using the actual distance.
'vote_distance_weights' written below does exactly that.
"""
def vote_distance_weights(neighbors, all_results = True) :
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors) :
        dist = neighbors[index][1]
        label = neighbors[index][2]
        class_counter[label] += 1 / (dist ** 2 + 1)
    labels, votes = zip(*class_counter.most_common())
    #print (labels, votes)
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results :
        total = sum(class_counter.values(), 0.0)
        for key in class_counter :
            class_counter[key] /= total
        return winner, class_counter.most_common()
    else :
        return winner, votes4winner / sum(votes)

for i in range(n_training_samples) :
    neighbors = get_neighbors(learnset_data, learnset_labels,
                              testset_data[i], 6, distance = distance)
    print("index: ", i, 
          ", result of vote: ", vote_distance_weights(neighbors, all_results = True) )

# To test the previous functions, we create another simple dataset

train_set = [(1, 2, 2), (-3, -2, 0), (1, 1, 3), (-3, -3, -1),
             (-3, -2, -0.5), (0, 0.3, 0.8), (-0.5, 0.6, 0.7),
             (0, 0, 0)]
labels = ['apple', 'banana', 'apple',
          'banana', 'apple', 'orange',
          'orange', 'orange']
k = 1
for test_instance in [(0, 0, 0), (2, 2, 2), (-3, -1, 0),
                      (0, 1, 0.9), (1, 1.5, 1.8), (0.9, 0.8, 1.6)] :
                      neighbors = get_neighbors(train_set, labels,
                                                test_instance, 2)
                      print("vote distance weights: ",
                            vote_distance_weights(neighbors))

"""
Next example comes from computer linguistics. We show how to use
kNN classifier to recognize misspelled words.
'levenshtein' module is used.
"""

from levenshtein import levenshtein

cities = []
with open("data/city_names.txt") as fh:
    for line in fh :
        city = line.strip()
        if " " in city:
            cities.append(city.split()[0])
        cities.append(city)
        # cities = cities[:20]
for city in ["Freiburg", "Frieburg", "Freiborg",
             "Hamborg", "Sahrluis"] :
             neighbors = get_neighbors(cities, cities,
                                       city, 2, distance = levenshtein)
             print("vote_distance_weights: ", vote_distance_weights(neighbors))

"""
We use extremely misspelled words in the following example. 
We see that our simple vote_prob function is doing well only in two cases: 
In correcting "holpposs" to "helpless" and "blagrufoo" to "barefoot". 
Whereas our distance voting is doing well in all cases. 
Okay, we have to admit that we had "liberty" in mind, when we wrote "liberdi", 
but suggesting "liberal" is a good choice.
"""

words = []
# os.chdir("working directory containing british-english.txt")
with open("british-english.txt") as fh:
    for line in fh:
        word = line.strip()
        words.append(word)
for word in ["holpful", "kundnoss", "holpposs", "blagrufoo", "liberdi"] :
    neighbors = get_neighbors(words, words, word, 3, distance = levenshtein)
    print("vote_distance_weights: ", vote_distance_weights(neighbors, all_results = False))
    print("vote_prob: ", vote_prob(neighbors))

"""
Example below implements kNN using sklearn.
'KNeighborsClassifier' kNN classifier from 'sklearn.neighbors'
is used on Iris data set
"""

# Creating and fitting a kNN classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(learnset_data, learnset_labels)
KNeighborsClassifier(algorithm = 'auto', 
                     leaf_size = 30,
                     metric = 'minkowski',
                     metric_params = None,
                     n_jobs = 1, 
                     n_neighbors = 5,
                     p = 2,
                     weights = 'uniform')
print("Predictions from the classifier: ")
print(knn.predict(testset_data))
print("Target values: ")
print(testset_labels)

learnset_data[:5], learnset_labels[:5]

# We again see that the sample with index 8 is not recognized properly.