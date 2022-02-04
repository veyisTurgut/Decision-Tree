"""
Adalet Veyis Turgut - 2017400210
"""

import csv
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import random

SEPARATORS = []
# [[0]: 0 for vertical, 1 for horizantal. [1]: boundary value. [2]:[start,finish] of line, [3]:[[w_1,E_1],[w_2,E_2]], [4]: level ]


def plotData(data):
    """
    Plots the given data points and colors them accooding to their labels.
    Also draws boundaries with decreasing thicknesses and different colors. 
    """
    data1 = [point[0] for point in data]
    data1_for_color = [point[1] for point in data]
    plt.figure()
    plt.scatter([x[0] for x in data1], [x[1] for x in data1], c=data1_for_color, vmin=min(
        data1_for_color), vmax=max(data1_for_color), cmap='Set3')

    colors = "rgbmc"
    for seperator_info in sorted(SEPARATORS, key=lambda x: x[4]):
        """
        seperator_info[0]: 0 for vertical line, 1 for horizantal line
        seperator_info[1]: boundary value
        seperator_info[2]: [0]: boundary start value, [1]: boundary finish value
        seperator_info[3]: [0]: [W_left, E_left] or [W_lower, E_lower], [1]: [W_right, E_right] or [W_upper, E_upper]
        seperator_info[4]: boundary level
        """
        if seperator_info[0] == 0:
            plt.vlines(x=seperator_info[1], ymin=seperator_info[2][0], ymax=seperator_info[2][1], colors=colors[seperator_info[4] % 5],
                       lw=4-seperator_info[4]*0.5, label="\nBoundary " + str(seperator_info[4])+"\nX = "+str(seperator_info[1]) +
                       "\nE_left = "+str(seperator_info[3][0][1])+"\nE_right = "+str(seperator_info[3][1][1]) +
                       "\nE_total = "+str(seperator_info[3][1][1]*seperator_info[3][1][0] + seperator_info[3][0][1]*seperator_info[3][0][0]))
        else:
            plt.hlines(y=seperator_info[1], xmin=seperator_info[2][0], xmax=seperator_info[2][1], colors=colors[seperator_info[4] % 5],
                       lw=4-seperator_info[4]*0.5, label="\nBoundary "+str(seperator_info[4]) + "\nY = "+str(seperator_info[1]) +
                       "\nE_lower = "+str(seperator_info[3][0][1])+"\nE_upper = "+str(seperator_info[3][1][1]) +
                       "\nE_total = "+str(seperator_info[3][1][1]*seperator_info[3][1][0] + seperator_info[3][0][1]*seperator_info[3][0][0]))
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
    plt.show()


def readData(path):
    """
    reads the csv file in the given path
    """
    file = open(path)
    reader = csv.reader(file)
    return [[(float(x), float(y)), int(label)] for (x, y, label) in reader]


def myData():
    """
    Generates simple dataset with complex decision boundaries
    """
    random.seed(10)  # make seed 10
    data = []
    for _ in range(300):  # generate 300 data points
        x = random.random()*6  # x values are between 0,6
        y = random.random()*6  # y values are between 0,6
        if (x < 4 and y > 5) or (x > 4 and y < 2) or (2 < x and x < 4 and y < 1):
            # label as 1 if the condition above is true
            data.append([(x, y), 1])
        else:
            data.append([(x, y), 0])
    return data


def entropy(data):
    """
    Returns the entropy of the given data.

    formula: -1 * sum (Pi * log(Pi)) for each label i
    data: [[(x,y), label]]
    """
    label_counts = {label: sum(map(lambda x: x[1] == label, data)) for label in set([x[1] for x in data])}
    return - sum(label_counts[label]/len(data)*math.log(label_counts[label]/len(data), 2) for label in label_counts)


def informationGain(entropy_data, data, subdatas):
    """
    Returns the information gain of a possible split and [Weight, entropy] pair of splittes subdatas

    formula: Entropy(data) - sum(Entropy(Si))*|Si| / |Data| for each label i
    entropy_data: total entropy of the given data
    data: [[(x,y), label]]
    subdata: [[(x,y), label]], splitted data as (left and right) or (lower and upper)
    """
    return entropy_data - sum(entropy(subdata)*len(subdata)/len(data) for subdata in subdatas), [[len(subdata)/len(data), entropy(subdata)] for subdata in subdatas]


def splitDataset(data, level, maxlevel):
    """
    This is a recursive algorithm.
    At each consecutive call, the level parameter is increased by 1 to indicate current depth. 
    There are two stop conditions: when depth(level) is equals to the maxlevel or when splitted subdatas are pure.
    Possible split points are the datapoints. I chose this approach instead of taking the averages of two consecutive data points.
    The point where information gain is maximum is decided as the split point.
    """
    if level == maxlevel:
        return
    best_split, index, dimension, entropies_l_r_or_u_l = -100, -1, -1, [] # initialize the variables
    # calculate entropy
    current_entropy = entropy(data)
    ####### split of x
    # sort data with respect to x axixes of points
    data.sort(key=lambda x: x[0][0])
    for i in range(len(data)):
        # for each data point, assume that it is the split point and calculate information gain
        i_g, entropies = informationGain(current_entropy, data, [data[:i], data[i:]])
        if i_g > best_split:
            # store the information of the point where information gain is maximum
            best_split, index, dimension, entropies_l_r_or_u_l = i_g, i, 0, entropies
    ####### split of y
    # copy data since we will sort it according to y axixes
    temp_data = data.copy()
    # sort data with respect to y axixes of points
    temp_data.sort(key=lambda x: x[0][1])
    for i in range(len(temp_data)):
        # for each data point, assume that it is the split point and calculate information gain
        i_g, entropies = informationGain(current_entropy, temp_data, [temp_data[:i], temp_data[i:]])
        if i_g > best_split:
            # store the information of the point where information gain is maximum
            best_split, index, dimension, entropies_l_r_or_u_l = i_g, i, 1, entropies
    if dimension == 0:
        ##### vertical split
        # boundary line was vertical, store its information in the global SEPARATOR variable.
        seperator_info = [0, data[index][0][0], [min(map(lambda point:point[0][1], data)), max(
            map(lambda point:point[0][1], data))], entropies_l_r_or_u_l, level+1]
        SEPARATORS.append(seperator_info)
        # calculate entropies of subdatas
        entropy_of_right = entropy(data[index:])
        entropy_of_left = entropy(data[:index])
        if entropy_of_left == entropy_of_right == 0:
            # perfect split, both of he subdatas are pure
            return
        elif entropy_of_left < 0.01:
            # if left part is almost pure, continue with right part
            splitDataset(data[index:], level+1, maxlevel)
        elif entropy_of_right < 0.01:
            # if right part is almost pure, continue with left part
            splitDataset(data[:index], level+1, maxlevel)
        else:
            # if none of the subdatas are pure, split them again
            splitDataset(data[index:], level+1, maxlevel)
            splitDataset(data[:index], level+1, maxlevel)
    else:
        ##### horizantal split
        # boundary line was vertical, store its information in the global SEPARATOR variable.
        seperator_info = [1, temp_data[index][0][1], [min(map(lambda point:point[0][0], temp_data)), max(
            map(lambda point:point[0][0], temp_data))], entropies_l_r_or_u_l, level+1]
        SEPARATORS.append(seperator_info)
        # calculate entropies of subdatas
        entropy_of_upper = entropy(temp_data[index:])
        entropy_of_lower = entropy(temp_data[:index])
        if entropy_of_lower == entropy_of_upper == 0:
            # perfect split, both of he subdatas are pure
            return
        elif entropy_of_lower < 0.01:
            # if lower part is almost pure, continue with upper part
            splitDataset(temp_data[index:], level+1, maxlevel)
        elif entropy_of_upper < 0.01:
            # if upper part is almost pure, continue with lower part
            splitDataset(temp_data[:index], level+1, maxlevel)
        else:
            # if none of the subdatas are pure, split them again
            splitDataset(temp_data[index:], level+1, maxlevel)
            splitDataset(temp_data[:index], level+1, maxlevel)

    if level == 0:
        # when we are finished and about to return to the main, plot the dataset with boundaries
        plotData(data)


def skLearn(data, max_depth):
    plt.figure()
    # set criterion to "entropy" since default mode is "gini". First parameter is data points, second parameter is labels
    tree = DecisionTreeClassifier(criterion='entropy').fit([x[0] for x in data], [x[1] for x in data])
    plot_tree(tree, filled=True, max_depth=max_depth)
    plt.show()


if __name__ == "__main__":
    # DEFAULT DATASET
    DATA = readData("data.txt")
    # MY ALGO
    splitDataset(data=DATA, level=0, maxlevel=2)
    # SK-LEARN
    skLearn(data = DATA, max_depth=2)

    # MY DATASET
    DATA = myData()
    SEPARATORS.clear()
    # MY ALGO
    splitDataset(data=DATA, level=0, maxlevel=10)
    # SK-LEARN
    skLearn(data =DATA, max_depth=10)
