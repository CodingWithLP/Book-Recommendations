import pandas as pd
from sklearn import tree
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import warnings
import pickle
from collections import Counter

df = pd.read_csv("C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\ML\\HackJPS\\book_levels.csv")
df = df.dropna()

levels = {'level_6a': 0, 'level_5a':0, 'level_4a': random.randint(0,1), 'level_3a': 1, 'level_A1': 1, 'level_A2': 1, 'level_B1': 2, 'level_B2': 2, 'level_C1': 2, 'level_C2': 3, 'level_D':3, 'level_E':4, 'level_F':random.randint(4,5), 'level_G':5, 'level_H':6, 'level_I': 7, 'level_J':7, 'level_K': 8, 'level_L': 9}
df['Language Level'] = df['Language Level'].map(levels)

# l = df.values.tolist()
# with open('C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\ML\\HackJPS\\userinput.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Name', 'Author', 'Language Level', 'Fit'])
#     for i in range(15):
#         t = random.randint(320)
#         row = l[t]
#         im = "Is", row[0], 'made by', row[1], "a book that would fit in your reading level (Enter a 1 for yes and 0 for no)?\n"
#         try:
#             inp = int(input(im))
#         except ValueError:
#             inp = int(input("Sorry, try again with the number 1 or 0: \n"))
#         if inp == 0 or inp == 1:
#             row.append(inp)
#             writer.writerow(row)


usdf = pd.read_csv("C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\ML\\HackJPS\\userinput.csv")
usdf = usdf.dropna()

X = usdf['Language Level']

y = usdf['Fit']

dtree = DecisionTreeClassifier()
dtree = dtree.fit([X], [y])

with open('spmodel.pickle', 'wb') as f:
    pickle.dump(dtree, f)

