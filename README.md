# CSE40647/CSE60647 - HW2

This is the second assignment for my Data Science Class.

## Getting Started

This project uses Python 3.

Clone this repo with the following command.

```
git clone https://github.com/ptins/HW1-DataScience.git
```

## Task

Given Notre Dame’s football game data for the last two seasons (2015 and 2016), can we
construct three classification models to predict game results on games in 2017? Can we evaluate
the model performance? The three classification models are ID3, C4.5, and Naïve Bayes.

## Decision Trees

The below function builds either an ID3 or C4.5 tree depending on the 'crit' argument.

```
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
```

First we need to split the data into training and testing sets.

```
train = pd.read_csv('./train.txt', sep='\t', index_col='ID').drop(['Opponent','Date'],axis=1)
test  = pd.read_csv('./test.txt', sep='\t', index_col='ID').drop(['Opponent','Date'],axis=1)
X = train.drop('Label', axis=1)
y = train.Label
```

Then we build the trees.

```
tree_id3 = build_tree(X, y, 'ID3')
tree_c45 = build_tree(X, y, 'C4.5')
```

To evaluate the models, run the test set predictions through the below function.

```
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
```

## Naive Bayes

The below function predicts the 'Label' feature for the testing set.

```
def naive_bayes(train, test_row):
    
    # split up dataset into winning and losing
    winning_set, losing_set = train[train['Label']=='Win'], train[train['Label']=='Lose']
    
    # for overall set, compute p(y=win)
    p_y = len(winning_set)/len(train)
    
    # for winning set, compute conditional probabilities
    conditional_probs = []
    for col in train.drop('Label', axis=1).columns:
        conditional_probs.append(len(winning_set[winning_set[col]==test_row[col]]) / len(winning_set))
    
    # compute p(y=win|x1,x2,x3) 
    win = np.prod(np.array(conditional_probs)) * p_y
    
    # for overall set, compute p(y=lose)
    p_not_y = len(losing_set)/len(train)

    # for winning set, compute conditional probabilities
    conditional_probs = []
    for col in train.drop('Label', axis=1).columns:
        conditional_probs.append(len(losing_set[losing_set[col]==test_row[col]]) / len(losing_set))
    
    # compute p(y=lose|x1,x2,x3)
    lose = np.prod(np.array(conditional_probs)) * p_not_y
         
    if win>=lose:
        return 'Win'
    else:
        return 'Lose'
```

We can evaluate the model by using the evaluation_metrics function above.

## Conclusions

The Naive Bayes algorithm performs the least well, while the C4.5 outperforms the ID3 decision tree. The lack of predictive power in the Naive Bayes case comes from the assumption that all features are conditionally independent.
