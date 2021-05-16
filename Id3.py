# instalations:
import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

eps = np.finfo(float).eps
from numpy import log2 as log


####################part 1#########################

# A vertex in a decision tree. Each node divides the example, by an f value and a t value.
# For each node if its f value is bigger than t_i, it'll go right, otherwise it'll go left.
# If this node is a leaf, the example will receive its status, either 'M' or 'B'.
class Node:
    f = None
    t_i = None
    left_child = None
    right_child = None
    status = None

    def __init__(self, f=None, t_i=None, left_child=None, right_child=None, status=None):
        self.f = f
        self.t_i = t_i
        self.left_child = left_child
        self.right_child = right_child
        self.status = status

# decision tree
class Tree:
    root = Node()

    def __init__(self, node):
        self.root = node

    def give_root(self):
        return self.root

# Given a group of Examples E and a group of features F, find for each single feature "f" in F -
# the amount of entropy they add, the subgroups they divide E into and the "t" values which might be selected for each f
def IG_Continuous(E, F):
    entropy_dict = {}
    subgroups1_dict = {}
    subgroups2_dict = {}
    t_is_dict = {}
    d = 0
    for f in F:
        entropy_dict[f] = {}
        subgroups1_dict[f] = {}
        subgroups2_dict[f] = {}
        values = []
        E_healthy = E[E['diagnosis'] == 'B'].shape[0]
        E_p = E_healthy/E.shape[0]
        E_entropy = -(E_p + eps)*log(E_p + eps)
        for i in range(E.shape[0]):
            values.append(E[f][i])
        values.sort()
        t_is = []
        if (len(values) == 1):
            t_is.append(values[0])
        for k in range(len(values) - 1):
            t_is.append(((values[k] + values[k + 1]) / 2))
        t_is_dict[f] = t_is
        for t_i in t_is:
            sub_groups_1 = E[E[f] >= t_i].reset_index(drop=True)
            group_1_size = sub_groups_1.shape[0]
            group_1_healthy = sub_groups_1[sub_groups_1['diagnosis'] == 'B'].shape[0]
            fraction_1 = 1
            if(group_1_size != 0):
                fraction_1 = group_1_healthy / group_1_size
            sub_groups_2 = E[E[f] < t_i].reset_index(drop=True)
            group_2_size = sub_groups_2.shape[0]
            group_2_healthy = sub_groups_2[sub_groups_2['diagnosis'] == 'B'].shape[0]
            fraction_2 = 1
            if(group_2_size != 0):
                fraction_2 = group_2_healthy / group_2_size
            sub_entropy = (group_1_size/E.shape[0])*(-(fraction_1 + eps) * np.log(fraction_1 + eps)) + (
                        (group_2_size/E.shape[0])*(-(fraction_2 + eps) * np.log(fraction_2 + eps)))
            entropy_dict[f][t_i] = E_entropy - sub_entropy
            subgroups1_dict[f][t_i] = sub_groups_1
            subgroups2_dict[f][t_i] = sub_groups_2
    return entropy_dict, subgroups1_dict, subgroups2_dict, t_is_dict


# Given the possible entropy additions which we calculated for each single feature "f" in F, we'll return
# the  the biggest entropy addition for one of the feature, we'll return that feature, the "t" value
# to go along with it and the subgroups created in E by this division
def IG_Continuous_aux(E, F):
    temp_F = F.copy()
    if 'diagnosis' in temp_F:
        temp_F.remove('diagnosis')
    entropy_dict, subgroups1_dict, subgroups2_dict, t_is_dict = IG_Continuous(E, temp_F)
    max_entropy = (-1) * (math.inf)
    max_f = (-1) * (math.inf)
    max_t_i = (-1) * (math.inf)
    for f in temp_F:
        for t_i in t_is_dict[f]:
            if (entropy_dict[f][t_i] > max_entropy):
                max_entropy = entropy_dict[f][t_i]
                max_f = f
                max_t_i = t_i
    return max_entropy, max_f, max_t_i, subgroups1_dict[max_f][max_t_i], subgroups2_dict[max_f][max_t_i]


# Given a group of diagnosises, see which is the majority diagnosis. Note: in this specific function E isn't a DataFrame
# or a list of all the examples like in other function, but a list of diagnosises
def MajorityClass(E):
    n = len(E)

    num_M = sum(1 for i in E if i == 'M')
    if (num_M > (n / 2)):
        if (num_M == n):
            return 'M', True
        else:
            return 'M', False
    else:
        if (num_M == 0):
            return 'B', True
        else:
            return 'B', False

# Recursive function to built the nodes (vertices) in our decision tree, based on a group of examples (E),
# a list of features (F) and a default verdict passed down from the parent node
def TDIDT(E, F, Default):
    default_node = Node(None, None, None, None, Default)
    if (E.shape[0] == 0):
        return default_node
    diags = E['diagnosis'].tolist()
    c, absolute_class = MajorityClass(diags)
    if (len(F) == 0 or absolute_class):
        return Node(None, None, None, None, c)
    entropy, f, t_i, sub_group1, sub_group2 = IG_Continuous_aux(E, F)
    return Node(f, t_i, TDIDT(sub_group2, F, c), TDIDT(sub_group1, F, c), c)


# A function to build a decision tree based on the "ID3" algorithm
def ID3(E, F):
    diags = E['diagnosis'].copy().tolist()
    c, b1 = MajorityClass(diags)
    tree1 = Tree(TDIDT(E, F, c))
    return tree1

# An aux-enveloping function to ID3
def ID3_train_build_tree(train_data):
    E = train_data.copy()
    F = list(train_data.copy().keys())
    tree1 = ID3(E, F)
    return tree1

# Recursive function, that works on one example and one node at a time, in order to get to the right leaf
# and give E its diagnosis
def test_one_item(E, node: Node):
    if (node.f == None):
        return node.status
    if (E[node.f][0] < node.t_i):
        return test_one_item(E, node.left_child)
    else:
        return test_one_item(E, node.right_child)

# Given a group of examples E and a decision tree, return a list containing side by side, the prediction made
# by the decision tree next to the example's real diagnosis
def test(E, tree):
    list_of_predictions = {}
    F = list(E.copy().keys())
    i = 0
    E_values = {}
    for i in range(E.shape[0]):
        E_values[i] = {}
        for f in F:
            E_values[i][f] = [E.iloc[i][f]]
    for i in range(E.shape[0]):
        temp_df = pd.DataFrame(E_values[i])
        list_of_predictions[i] = {}
        list_of_predictions[i][0] = (test_one_item(temp_df, tree.root))
        list_of_predictions[i][1] = (E.iloc[i]['diagnosis'])
    return list_of_predictions

# An aux-enveloping function for "test" which after recieving the prediction list, records the percentage of errors
# in prediction out of all the prediction and returns it
def test_aux(test_data,tree):
    E = test_data.copy()
    comparison_list = test(E, tree)
    count = 0
    i = 0
    for e in E.iterrows():
        if (comparison_list[i][0] == comparison_list[i][1]):
            count += 1
        i += 1
    den = len(comparison_list)
    return (count / den)

# A function to create the relevant dataframes from the training and testing files. Relevant for more questions, not just
# question 1
def data_q1(train_file,test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    return train,test

####################Question 3#########################

# Similar to the TDIDT built for question one but with pruning dependent on parameter M.
# If number of examples left is smaller than M, create leaf.
# Built for question 3, but used for more questions
def TDIDT_M_pruning(E, F, Default,M):
    default_node = Node(None, None, None, None, Default)
    if (E.shape[0] == 0):
        return default_node
    diags = E['diagnosis'].tolist()
    c, absolute_class = MajorityClass(diags)
    if (len(F) == 0 or absolute_class or E.shape[0]<=M):
        return Node(None, None, None, None, c)
    entropy, f, t_i, sub_group1, sub_group2 = IG_Continuous_aux(E, F)
    return Node(f, t_i, TDIDT_M_pruning(sub_group2, F, c,M), TDIDT_M_pruning(sub_group1, F, c,M), c)

# Similar to the basic ID3 function built for question 1, but adjusted to M pruning.
# This function was built for question 3, but is used for other questions as well.
def ID3_M_pruning(E, F, M):
    diags = E['diagnosis'].copy().tolist()
    c, b1 = MajorityClass(diags)
    tree1 = Tree(TDIDT_M_pruning(E,F,c,M))
    return tree1

# Similar to the basic ID3-aux function built for question 1, but adjusted to M pruning.
# This function was built for question 3, but is used for other questions as well.
def ID3_train_build_tree_M_pruning(train_data,M):
    E = train_data.copy()
    F = list(train_data.copy().keys())
    tree1 = ID3_M_pruning(E,F,M)
    return tree1

# Using the K-Fold algorithm, we'll create different decision trees using different parts of the training data.
# We'll test each using different parts of the training data and give the average result for those tests
def train_and_test_q3(train_file, M_values):
    E = pd.read_csv(train_file)
    kf = KFold(n_splits = 5,shuffle = True,random_state=205703853)
    results = {}
    avg = []
    for M in M_values:
        results[M] = []
        for train_index, test_index in kf.split(E):
            train_dict = {}
            test_dict = {}
            F = list(E.copy().keys())
            for f in F:
                train_dict[f] = []
                test_dict[f] = []
            for i in train_index:
                for f in F:
                    train_dict[f].append(E.iloc[i][f])
            for i in test_index:
                for f in F:
                    test_dict[f].append(E.iloc[i][f])
            train_data_frame = pd.DataFrame.from_dict(train_dict)
            test_data_frame = pd.DataFrame.from_dict(test_dict)
            tree1 = ID3_train_build_tree_M_pruning(train_data_frame,M)
            results[M].append(test_aux(test_data_frame,tree1))
        avg.append(sum(results[M])/5)
    return avg


####################Question 4#########################

# The test-aux built especially for question 4. Similar to the function used in question 1, but returns loss instead
# of "success rate". Built for question 4, but used for other questions as well.
def test_aux_q4(test_data,tree):
    E = test_data.copy()
    comparison_list = test(E, tree)
    count = 0
    for i in range(E.shape[0]):
        if (comparison_list[i][0] != comparison_list[i][1]):
            if(comparison_list[i][0] == 'B'):
                count += 1
            else:
                count+=0.1
    den = len(comparison_list)
    return (count / den)

# The train and test tree built especially for question 4. Similar to the function used in question 3,
# but returns loss instead of "success rate".
# Also, as written in the word document, Trial with parts of the training data is necessary for my algorithm.
def train_and_test_q4(train_file, M_values):
    E = pd.read_csv(train_file)
    kf = KFold(n_splits = 5,shuffle = True,random_state=205703853)
    results = {}
    loss = []
    for M in M_values:
        results[M] = []
        for train_index, test_index in kf.split(E):
            train_dict = {}
            test_dict = {}
            F = list(E.copy().keys())
            for f in F:
                train_dict[f] = []
                test_dict[f] = []
            for i in train_index:
                for f in F:
                    train_dict[f].append(E.iloc[i][f])
            for i in test_index:
                for f in F:
                    test_dict[f].append(E.iloc[i][f])
            train_data_frame = pd.DataFrame.from_dict(train_dict)
            test_data_frame = pd.DataFrame.from_dict(test_dict)
            tree1 = ID3_train_build_tree_M_pruning(train_data_frame,M)
            results[M].append(test_aux_q4(test_data_frame,tree1))
        loss.append(sum(results[M])/5)
    return loss




def main():
    train,test = data_q1('train.csv','test.csv')
    tree1 = ID3_train_build_tree(train)
    percent = test_aux(test, tree1)
    print(percent)
    return 0



if __name__ == "__main__":
    main()
