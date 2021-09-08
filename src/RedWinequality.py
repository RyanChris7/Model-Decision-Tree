from __future__ import print_function
from csv import reader

def load_csv(filename):
    file = open(filename,"rt")
    lines = reader(file)
    dataset = list(lines)
    del dataset[0]
    return dataset
    
traindatadirectory = 'train.csv'
testdatadirectory = 'test.csv'
training_data = [[float(y) for y in x] for x in load_csv(traindatadirectory)]
testing_data = [[float(y) for y in x] for x in load_csv(testdatadirectory)]
header = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol","quality"]
# to load all the data and enable pyhton to read the data through reader

def quality_classifier(dataset):
     for i in range (len(dataset)):

        temp = dataset[i]

        for k in range(len(temp)):
            temp[k] = float(temp[k])
        
        if temp[-1] > 6:
            temp[-1] = 1
        else:
            temp[-1] = 0
#classify the quality of wine in each data set 
quality_classifier(training_data)
quality_classifier(testing_data)

def data_count(training_data):
    qualitydirectory = {0:0, 1:0}
    for k in training_data:
        if k[-1] == 0:
            qualitydirectory[0] += 1
        else:
            qualitydirectory[1] += 1
    return qualitydirectory

def values(rows, col):
    return set([row[col] for row in rows])
# find the values in each row and column on the dataset
def num_check(value):
    return isinstance(value, int) or isinstance(value, float)
#check if the value is integer or float

class Question:

    def __init__(self, column, value):   #initializer to generate teh column and value needed
        self.column = column
        self.value = value

    def match(self, example):     # function to compare the value and feauture example
        val = example[self.column]
        if num_check(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):      #method to print a readable fromat for the question
        condition = "=="
        if num_check(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):    #Check if the row is matched with the question and return trows or frows
    Trows, Frows = [], []
    for row in rows:
        if question.match(row):
            Trows.append(row)
        else:
            Frows.append(row)
    return Trows, Frows


def gini_index(rows):    #Calculate the gini impurity on list of rows
    counts = data_count(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):

    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini_index(left) - (1 - p) * gini_index(right)

def best_split(rows): #function to split tree into differnet nodes
    gain = 0 
    question_limit = None  
    current_uncertainty = gini_index(rows)
    n_features = len(rows[0]) - 1  

    for col in range(n_features): 

        values = set([row[col] for row in rows]) 

        for val in values: 

            question = Question(col, val)

            #to split the datset
            Trows, Frows = partition(rows, question)

            
            if len(Trows) == 0 or len(Frows) == 0:
                continue

            #calculate the infromation from the split
            gain = info_gain(Trows, Frows, current_uncertainty)


            if gain >= gain:
                gain, question_limit = gain, question

    return gain, question_limit


class Leaf:

    def __init__(self, rows):
        self.predictions = list(data_count(rows))[0]


class Decision_Node:

    def __init__(self,
                 question,
                 Tbranch,
                 Fbranch):
        self.question = question
        self.Tbranch = Tbranch
        self.Fbranch = Fbranch


def tree_function(rows):
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = best_split(rows)

    if gain == 0:
        return Leaf(rows)
    Trows, Frows = partition(rows, question)
    Tbranch = tree_function(Trows)

    Fbranch = tree_function(Frows)

    return Decision_Node(question, Tbranch, Fbranch)


def print_tree(node, spacing=""):    #print the tree from root to leaf

    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))
    print (spacing + '--> True:')
    print_tree(node.Tbranch, spacing + "  ")
    print (spacing + '--> False:')
    print_tree(node.Fbranch, spacing + "  ")


def classify(row, node):
    #the case have reach the leaf
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.Tbranch)
    else:                                       #to decide wheter to continue on fbranch or tbranch
        return classify(row, node.Fbranch)

if __name__ == '__main__':
#Run the code and print the result
    my_tree = tree_function(training_data)
    print_tree(my_tree)
    test_sample = 0
    correct_prediction=0
    
    for row in testing_data:
        prediction = classify(row,my_tree)
        real = (row[-1])
        #print('Real', real, '|', prediction,'Predict')
        if prediction == real:
            test_sample+=1
            correct_prediction+=1
        else:
            test_sample+=1 
    print("Results:")
    print("Total sample data: " ,test_sample)
    print("Total correct prediction: " ,correct_prediction)
    accuracy = (correct_prediction/test_sample)*100
    print("Accuracy: ",accuracy)