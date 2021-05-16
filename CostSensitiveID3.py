import Id3
# instalations:
import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

eps = np.finfo(float).eps
from numpy import log2 as log


####################Question 4#############################


# Similar to the regular node, but includes 2 more parameters: num_B and num_M
# They're used in the part where we test the tree with parts of the training file. For each leaf we'll count the amount
# of examples being given this node's status from each category (based on real status of the examples).
class CS_Node:
    f = None
    t_i = None
    left_child = None
    right_child = None
    status = None
    num_B = None
    num_M = None

    def __init__(self, f=None, t_i=None, left_child=None, right_child=None, status=None):
        self.f = f
        self.t_i = t_i
        self.left_child = left_child
        self.right_child = right_child
        self.status = status
        self.num_B = 0
        self.num_M = 0

# Recursive function which tries to get to all the leaves of the tree. Given that we count each "B-mistake" as 0.1
# of an "M-mistake", if the amount of "B" examples classified by a leaf is less than 10 times the number of "M"
# examples (classified as B), the function will flip a 'B' leaf into an 'M' leaf inorder to reduce the overall loss.
    def fix_node(self):
        if(self.left_child!=None):
            self.left_child.fix_node()
        if((self.f == None) and (self.status == 'B') and(0.1*(self.num_B)<self.num_M)):
            self.status = 'M'
        if (self.right_child != None):
            self.right_child.fix_node()

    def inc_num_B(self):
        self.num_B+=1

    def inc_num_M(self):
        self.num_M+=1

    def change_status_from_B_to_M(self):
        self.status = 'M'

class CS_Tree:
    root = CS_Node()

    def __init__(self, node):
        self.root = node

    def give_root(self):
        return self.root


# As described in the "fix-node" function, this function flips some "B" leaves into "M" leaves inorder to reduce the
# overall loss
    def fix_tree_aux(self):
        self.root.fix_node()


# Similar to the regular TDIDT, uses M pruning and uses CS Node instead of regular nodes
def CS_TDIDT(E, F, Default,M):
    default_node = CS_Node(None, None, None, None, Default)
    if (E.shape[0] == 0):
        return default_node
    diags = E['diagnosis'].tolist()
    c, absolute_class = Id3.MajorityClass(diags)
    if (len(F) == 0 or absolute_class or E.shape[0]<=M):
        return CS_Node(None, None, None, None, c)
    entropy, f, t_i, sub_group1, sub_group2 = Id3.IG_Continuous_aux(E, F)
    return CS_Node(f, t_i, CS_TDIDT(sub_group2,F,c,M), CS_TDIDT(sub_group1,F,c,M), c)

# Similar to the regular ID3, uses M pruning and uses a CS tree instead of a regular tree
def CS_ID3(E, F,M):
    diags = E['diagnosis'].copy().tolist()
    c, b1 = Id3.MajorityClass(diags)
    tree1 = CS_Tree(CS_TDIDT(E,F,c,M))
    return tree1

# Similar to the the regular ID3-aux, uses M pruning and uses a CS tree instead of a regular tree
def CS_ID3_train_build_tree(train_data,M):
    E = train_data.copy()
    F = list(train_data.copy().keys())
    tree1 = CS_ID3(E,F,M)
    return tree1

# Similar to the regular function to test one item, but also increments "num_B" and "num_M" in the tree's leaves
# as part of the algorithm
def CS_test_one_item(E, node):
    if (node.f == None):
        if(E['diagnosis'][0] == 'B'):
            node.inc_num_B()
        else:
            node.inc_num_M()
        return node.status
    if (E[node.f][0] < node.t_i):
        return CS_test_one_item(E, node.left_child)
    else:
        return CS_test_one_item(E, node.right_child)

# Similar to the regular test function, but also adjusts the number of "M" and "B" examples each leaf has encountered
def CS_test(E, tree):
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
        list_of_predictions[i][0] = (CS_test_one_item(temp_df, tree.root))
        list_of_predictions[i][1] = (E.iloc[i]['diagnosis'])
        i += 1
    return list_of_predictions

# Similar to the regular test function, but also adjusts the number of "M" and "B" examples each leaf has encountered
# as well as calculating "loss" instead of "success rate".
def CS_test_aux(test_data,tree):
    E = test_data.copy()
    comparison_list = CS_test(E, tree)
    count = 0
    for i in range(E.shape[0]):
        if (comparison_list[i][0] != comparison_list[i][1]):
            if(comparison_list[i][0] == 'B'):
                count += 1
            else:
                count+=0.1
    den = len(comparison_list)
    return (count / den)

# The train function for question 4, designed to build a Cost-Sensetive tree. As part of that training, we'll also test
# the different trees with different parts of the training file. Given that we count each "B-mistake" as 0.1
# of an "M-mistake", if the amount of "B" examples classified by a leaf is less than 10 times the number of "M"
# examples (classified as B), the function will flip a 'B' leaf into an 'M' leaf inorder to reduce the overall loss.
# We'll return the loss value for the testing stage in this function before and after fixing the trees, as well as the
# tree out of the 5 trees with the best results
def CS_train_and_test(train_file, M):
    E = pd.read_csv(train_file)
    kf = KFold(n_splits = 5,shuffle = True,random_state=205703853)
    results_before = []
    results_after = []
    F = list(E.copy().keys())
    tree_list = []
    test_list = []
    for train_index, test_index in kf.split(E):
        train_dict = {}
        test_dict = {}
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
        test_list.append(test_data_frame)
        tree1 = CS_ID3_train_build_tree(train_data_frame,M)
        results_before.append(CS_test_aux(test_data_frame,tree1))
        tree_list.append(tree1)
    i=0
    for tree1 in tree_list:
        tree1.fix_tree_aux()
        results_after.append(CS_test_aux(test_list[i], tree1))
        i+=1
    loss_before = sum(results_before)/5
    loss_after = sum(results_after)/5
    i_max = results_after.index(min(results_after))
    return loss_before,loss_after,tree_list[i_max]

def main():
    loss_before,loss_after,tree1 = CS_train_and_test('train.csv',11)
    print("before:")
    print(loss_before)
    print("after:")
    print(loss_after)
    train,test = Id3.data_q1('train.csv','test.csv')
    tree1.fix_tree_aux()
    new_loss = CS_test_aux(test, tree1)
    print(new_loss)
    return 0

if __name__ == "__main__":
    main()

