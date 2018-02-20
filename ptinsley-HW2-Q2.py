
# coding: utf-8

# ## Imports

# In[1]:


# imports
import math
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics import confusion_matrix


# ## Functions

# In[2]:


def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}


# In[3]:


def entropy(s):

    # start with zero
    result = 0

    # get unique values and counts of set
    val, counts = np.unique(s, return_counts=True)
    freqs = counts/len(s)

    for p in freqs:
        result -= p * np.log2(p)

    return result


# In[4]:


def information_gain(y, x):

    # start with unconditional entroy of class Y --> H(Y)
    result = entropy(y)

    # calculate conditional entropy of class Y given feature X_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts/len(x)

    # 'sum up' weighted average of conditional entropies for X_i taking values x_i --> H(Y|X_i)
    for p, v in zip(freqs, val):
        result -= p * entropy(y[x == v])

    return result


# In[5]:


def split_information(y, x):

    # start with zero
    result = 0

    val, counts = np.unique(x, return_counts=True)
    freqs = counts/len(x)

    # 'sum up'
    for freq in freqs:
        result -= freq*np.log2(freq)

    return result


# In[6]:


def safe_divide(n, d):
    return n / d if d else 0


# In[7]:


def is_pure(s):
    return len(set(s)) == 1


# In[8]:


def build_tree(x, y, crit):

    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # get attr that yields highest information gain
    gain = np.array([information_gain(y, x[col]) for col in x.columns.values])

    if crit=='ID3':
        selected_attr = np.argmax(gain)

    elif crit=='C4.5':
        split_info = np.array([split_information(y, x[col]) for col in x.columns.values])
        gain_ratio = [safe_divide(x,y) for x, y in zip(gain, split_info)]
        selected_attr = np.argmax(gain_ratio)

    else:
        return 'Invalid splitting criterion...'

    # return y if no gain
    if np.all(gain < 1e-6):
        return y

    # split data using the selected attribute
    sets = partition(x[x.columns.values[selected_attr]])

    # create dictionary to hold next subsets
    result = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        result["%s = %s" % (x.columns.values[selected_attr], k)] = build_tree(x_subset, y_subset, crit)

    return result


# In[9]:


def evaluation_metrics(predictions):

    # confusion matrix
    conf_mat = confusion_matrix(test['Label'], predictions)
    print('Confusion Matrix:\n{}'.format(conf_mat))

    # compute metrics from tn, fp, fn, tp
    tn, fp, fn, tp = conf_mat.ravel()
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(recall*precision)/(recall+precision)

    print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1 Score: {}'.format(accuracy, precision, recall, f1))


# ## Read in Data

# In[10]:


train = pd.read_csv('./train.txt', sep='\t', index_col='ID').drop(['Opponent','Date'],axis=1)
test  = pd.read_csv('./test.txt', sep='\t', index_col='ID').drop(['Opponent','Date'],axis=1)
X = train.drop('Label', axis=1)
y = train.Label


# ## Decision Trees (ID3) - Building

# In[20]:


# tree_id3 = build_tree(X, y, 'ID3')
# pprint(tree_id3)


# ## Decision Trees (ID3) - Evaluation

# In[12]:


# {
#     'Media = 1-NBC': {
#         'Is_Opponent_in_AP25_Preseason = In':
#             Win (by root node majority)
#         'Is_Opponent_in_AP25_Preseason = Out': {
#             'Is_Home_or_Away = Away':
#                 Win
#             'Is_Home_or_Away = Home':
#                 Win (by node majority)
#         }
#     },
#     'Media = 2-ESPN':
#         Win
#     'Media = 3-FOX':
#         Lose
#     'Media = 4-ABC': {
#         'Is_Opponent_in_AP25_Preseason = In':
#             Lose (node purity)
#         'Is_Opponent_in_AP25_Preseason = Out':
#             Win (by node majority)
#     },
#     'Media = 5-CBS':
#         Lose
# }

predictions_id3 = ['Win','Win','Win','Lose','Win','Win','Win','Win','Win','Lose','Win','Lose']
test['Predictions_id3'] = predictions_id3


# In[13]:


# evaluation_metrics(test['Predictions_id3'])


# ## Decision Trees (C4.5) - Building

# In[21]:


tree_c45 = build_tree(X, y, 'C4.5')
pprint(tree_c45)


# In[15]:


# {
#     'Is_Opponent_in_AP25_Preseason = In': {
#         'Is_Home_or_Away = Away':
#             Lose (node purity)
#         'Is_Home_or_Away = Home':
#             Win (by root node majority)
#     },
#     'Is_Opponent_in_AP25_Preseason = Out': {
#         'Media = 1-NBC': {
#             'Is_Home_or_Away = Away':
#                 Win
#             'Is_Home_or_Away = Home':
#                 Win (by node majority)
#         },

###       'Media = 3-FOX':
###           Win (by root node majority)

#         'Media = 2-ESPN':
#             Win
#         'Media = 4-ABC':
#             Win (by node majority)
#         'Media = 5-CBS':
#             Lose
#     }
# }

predictions_c45 = ['Win','Win','Win','Win','Win','Win','Win','Win','Win','Lose','Win','Lose']
test['Predictions_c45'] = predictions_c45


# In[16]:


evaluation_metrics(test['Predictions_c45'])
