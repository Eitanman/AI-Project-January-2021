import CostSensitiveID3
# instalations:
import numpy as np
import pandas as pd
import math
import random
import sklearn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

eps = np.finfo(float).eps
from numpy import log2 as log


##################Question 6######################

# Crate a new Dataframe for one examples, out of a Dataframe for multiple examples
def make_object_from_one_row(E,row):
    F = list(E.copy().keys())
    E_values = {}
    for f in F:
        E_values[f] = [E.iloc[row][f]]
    temp_df = pd.DataFrame(E_values)
    return temp_df

# Calculate centroid for a Dataframe containing multiple examples
def calculate_centroid(E):
    F = list(E.copy().keys())
    F_temp = F.copy()
    F_temp.remove('diagnosis')
    centroid = {}
    for f in F_temp:
        centroid[f] = [0]
    for i in range(E.shape[0]):
        for f in F_temp:
            centroid[f][0] += E.iloc[i][f]
    for f in F_temp:
        centroid[f][0] = centroid[f][0]/E.shape[0]
    data_frame = pd.DataFrame.from_dict(centroid)
    return data_frame

# A data-type containing a decision tree, the Dataframe containing the examples used to build the tree and a DataFrame
# containing the centroid for those examples
class Tree_Example_and_Centroid:
    tree = None
    Examples = None
    Centroid = None

    def __init__(self,train_data,M = 0):
        self.tree = CostSensitiveID3.Id3.ID3_train_build_tree_M_pruning(train_data,M)
        self.Examples = train_data.copy()
        self.Centroid = calculate_centroid(train_data)

    def get_centroid(self):
        return self.Centroid

    def get_tree(self):
        return self.tree

    def get_examples(self):
        return self.Examples

# A data-type containing a list of "Tree-Example and Centroid"s, the list of centroids for those specific decision
# trees and the parameter K, which is the number of trees deciding each example
class KNN_Forest:
    list_of_trees = None
    list_of_centroids = None
    K = None

    def __init__(self,list_of_trees = None,list_of_centroids = None,K = 0):
        self.list_of_trees = list_of_trees
        self.list_of_centroids = list_of_centroids
        self.K = K

    def get_list_of_trees(self):
        return self.list_of_trees

    def get_list_of_centroids(self):
        return self.list_of_centroids

    def get_K(self):
        return self.K

# Creates a new Data-Frame containing "p" percentage of the original examples
def E_Sub_group(E,p):
    n = E.shape[0]
    F = list(E.copy().keys())
    train_dict = {}
    if(n*p<1):
        N=math.ceil(n*p)
    else:
        N=math.floor(n*p)
    N_range = [k for k in range(N)]
    index = []
    for i in range(N):
        temp = random.choice(N_range)
        index.append(temp)
        N_range.remove(temp)
    index.sort()
    for f in F:
        train_dict[f] = []
    for i in index:
        for f in F:
            train_dict[f].append(E.iloc[i][f])
    train_data_frame = pd.DataFrame.from_dict(train_dict)
    return train_data_frame

# Calculates mew and sigma categories for a group of examples used to calculate "f_tag", as was learned in the class
def mew_and_sigma(E,f):
    m = E.shape[0]
    if(m == 1):
        return 0,1
    f_values = []
    for i in range(m):
        f_values.append(E.iloc[i][f])
    mew = sum(f_values)/m
    f_sigma_vals = []
    for i in range(m):
        f_sigma_vals.append(pow((f_values[i] - mew),2))
    sigma = math.sqrt(sum(f_sigma_vals)/(m-1))
    return mew, sigma

# Calculates an adjusted value for a feature "f" and a single object, relative to the rest of the examples in E
def f_tag(E,f,Object):
    mew_f,sigma_f = mew_and_sigma(E,f)
    f_tag_object = (Object.iloc[0][f] - mew_f)/sigma_f
    return f_tag_object

# Calculates the adjusted "distance" between two objects, relative to the adjusted value of each feature, as was
# learned in class
def adjusted_distance(Object_1,Object_2,Group_1):
    F = list(Object_1.copy().keys())
    temp_F = F.copy()
    temp_F.remove('diagnosis')
    lengths = []
    for f in temp_F:
        o_1 = f_tag(Group_1,f,Object_1)
        o_2 = f_tag(Group_1,f,Object_2)
        lengths.append(abs(pow(o_1,2) - pow(o_2,2)))
    return math.sqrt(sum(lengths))


# Build a new KNN-Forest using a set of examples and parameters K,N,p and M
def KNN_Forest_train_and_build(train_data,K,N,p,M):
    E = train_data.copy()
    list_of_trees = []
    list_of_centroids = []
    for x in range(N):
        temp_sub_group = E_Sub_group(E, p)
        temp_Tree_Example_and_Centroid = Tree_Example_and_Centroid(temp_sub_group,M)
        list_of_trees.append(temp_Tree_Example_and_Centroid)
        list_of_centroids.append(temp_Tree_Example_and_Centroid.get_centroid())
    return KNN_Forest(list_of_trees,list_of_centroids,K)


#Given a single examples and a KNN-Forest choose K trees whose centroids are closest to the example with their adjusted
# values
def choose_K_trees(E,KNN_Forest):
    distances = []
    for i in range(len(KNN_Forest.get_list_of_trees())):
        temp_centroid = KNN_Forest.list_of_trees[i].get_centroid()
        temp_df = KNN_Forest.get_list_of_trees()[i].get_examples()
        distances.append(adjusted_distance(E,temp_centroid,temp_df))
    indexes = []
    for i in range(KNN_Forest.get_K()):
        temp_min = min(distances)
        indexes.append(distances.index(temp_min))
        distances.remove(temp_min)
    return indexes

# Given a group of verdicts by K trees, give us the majority opinion
def get_majority(group):
    num_M = sum(1 for i in group if i == 'M')
    num_B = sum(1 for i in group if i == 'B')
    if(num_B>num_M):
        return 'B'
    else:
        return 'M'

# A function to test one item built for question 6.
# For each example, choose K trees whose adjusted values are closest to it, let each tree give its prediction, then
# give a prediction according to the majority opinion.
def test_one_item_q6_aux(E, KNN_Forest):
    indexes = choose_K_trees(E,KNN_Forest)
    verdicts = []
    for i in indexes:
        verdicts.append(CostSensitiveID3.Id3.test_one_item(E,KNN_Forest.get_list_of_trees()[i].get_tree().root))
    return get_majority(verdicts)

# Similar to the regular "test" function, but uses KNN-Forest instead of a decision tree.
# Returns a prediction list, containing both the real diagnosis for each example and the KNN-Forest prediction.
def test_q6(E, KNN_Forest):
    list_of_predictions = {}
    for i in range(E.shape[0]):
        temp_df = make_object_from_one_row(E,i)
        list_of_predictions[i] = {}
        list_of_predictions[i][0] = (test_one_item_q6_aux(temp_df, KNN_Forest))
        list_of_predictions[i][1] = (E.iloc[i]['diagnosis'])
    return list_of_predictions

# Test-aux function designed for question 6. Similar to the functions in previous questions. Returns "loss" value.
def test_aux_q6(test_data,KNN_Forest):
    E = test_data.copy()
    comparison_list = test_q6(E, KNN_Forest)
    count = 0
    for i in range(E.shape[0]):
        if (comparison_list[i][0] != comparison_list[i][1]):
            if(comparison_list[i][0] == 'B'):
                count += 1
            else:
                count+=0.1
    den = len(comparison_list)
    return (count / den)




############################Question 7##########################

# Similar to the "adjusted distance" function for question 6, only this one doesn't calculate "distance" according
# to all features, only to a group of features, which are included in F.
def Improved_adjusted_distance(Object_1,Object_2,Group_1,F):
    temp_F = F.copy()
    lengths = []
    for f in temp_F:
        o_1 = f_tag(Group_1,f,Object_1)
        o_2 = f_tag(Group_1,f,Object_2)
        lengths.append(abs(pow(o_1,2) - pow(o_2,2)))
    return math.sqrt(sum(lengths))

# Choose a group of 4 features out of F, who will divide E in such a way that will add the most entropy
# as calculated by "IG_continuous_aux"
def Choose_4_best_features(E,F):
    temp_F = F.copy()
    best_features = []
    for i in range(4):
        entropy, f, t_i, sub_group1, sub_group2 = CostSensitiveID3.Id3.IG_Continuous_aux(E, temp_F)
        best_features.append(f)
        temp_F.remove(f)
    return best_features

# Choose K trees whose centroids are "closest" to an example "E", but with an "adjusted distance" calculated using
# the most indicative 4 features, not all the features.
def Improved_choose_K_trees(E,KNN_Forest):
    distances = []
    F = list(E.copy().keys())
    F.remove('diagnosis')
    temp_F = Choose_4_best_features(E,F)
    for i in range(len(KNN_Forest.get_list_of_trees())):
        temp_centroid = KNN_Forest.list_of_trees[i].get_centroid()
        temp_df = KNN_Forest.get_list_of_trees()[i].get_examples()
        distances.append(Improved_adjusted_distance(E,temp_centroid,temp_df,temp_F))
    indexes = []
    distances_copy = distances
    for i in range(KNN_Forest.get_K()):
        temp_min = min(distances)
        indexes.append(distances.index(temp_min))
        distances.remove(temp_min)
    return indexes,distances_copy

# Calculate the majority opinion of the K trees about a given example, only each tree is given an adjusted weight to
# its vote. The closer its centroid is to the example, the bigger its weight.
def Improved_get_majority(verdicts,distances):
    num_M = 0
    num_B = 0
    sum_distances = sum(distances[i] for i in range(len(distances)))
    for i in range(len(verdicts)):
        if(verdicts[i] == 'M'):
            num_M += 1 - (distances[i])/(sum_distances+eps)
        else:
            num_B += 1 - (distances[i])/(sum_distances+eps)
    if(num_B>num_M):
        return 'B'
    else:
        return 'M'

# Test one items using the function to choose K trees developed for question 7
def test_one_item_q7_aux(E, KNN_Forest):
    indexes,distances = Improved_choose_K_trees(E,KNN_Forest)
    verdicts = []
    for i in indexes:
        verdicts.append(CostSensitiveID3.Id3.test_one_item(E,KNN_Forest.get_list_of_trees()[i].get_tree().root))
    return Improved_get_majority(verdicts,distances)

# Test a set of examples using the function using the single-item test function developed for question 7
def test_q7(E, KNN_Forest):
    list_of_predictions = {}
    for i in range(E.shape[0]):
        temp_df = make_object_from_one_row(E,i)
        list_of_predictions[i] = {}
        list_of_predictions[i][0] = (test_one_item_q7_aux(temp_df, KNN_Forest))
        list_of_predictions[i][1] = (E.iloc[i]['diagnosis'])
    return list_of_predictions

# Test a set of examples using the function using the test function developed for question 7
def test_aux_q7(test_data,KNN_Forest):
    E = test_data.copy()
    comparison_list = test_q7(E, KNN_Forest)
    count = 0
    for i in range(E.shape[0]):
        if (comparison_list[i][0] != comparison_list[i][1]):
            if(comparison_list[i][0] == 'B'):
                count += 1
            else:
                count+=0.1
    den = len(comparison_list)
    return (count / den)


def main():
    train,test = CostSensitiveID3.Id3.data_q1('train.csv','test.csv')
    temp_KNN_Forest = KNN_Forest_train_and_build(train, 5, 10, 0.7, 11)
    loss = test_aux_q7(test, temp_KNN_Forest)
    print(loss)
    return 0


if __name__ == "__main__":
    main()
