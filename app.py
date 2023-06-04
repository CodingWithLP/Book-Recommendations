import numpy as np
from numpy import random
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import csv


app = Flask(__name__)

df = pd.read_csv("C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\ML\\HackJPS\\book_levels.csv")
model = pickle.load(open("C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\ML\\HackJPS\\spmodel.pickle", 'rb'))
html = "C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\ML\\HackJPS\\index.html"

levels = {'level_6a': 0, 'level_5a':0, 'level_4a': random.randint(0,1), 'level_3a': 1, 'level_A1': 1, 'level_A2': 1, 'level_B1': 2, 'level_B2': 2, 'level_C1': 2, 'level_C2': 3, 'level_D':3, 'level_E':4, 'level_F':random.randint(4,5), 'level_G':5, 'level_H':6, 'level_I': 7, 'level_J':7, 'level_K': 8, 'level_L': 9}
df['Language Level'] = df['Language Level'].map(levels)

l = df.values.tolist()

books = [['Hop on Pop', 'Dr. Seuss', 0],
    ['To Kill a Mockingbird', 'Harper Lee', 7],
    ['Old Black Fly', 'Jim Aylesworth', 0],
    ["Fever, 1793", 'Laurie Halse Anderson', 6],
    ['The Giver', 'Lois Lowry', 5],
    ['Jumanji', 'Chris Van Allsburg', 3],
    ['Green Eggs and Ham', 'Dr. Seuss', 0],
    ['The Magic School Bus: Inside the Earth', 'Joanna Cole and Bruce Degen', 1],
    ['The Mitten', 'Alvin Tresselt', 2],
    ['Flat Stanley: His Original Adventure', 'Jeff Brown', 2],
    ['Charlottes Web', 'E.B. White', 3],
    ['Lord of the Flies', 'William Golding', 9],
    ['Hamlet', 'William Shakespeare', 8],
    ['Charlie and the Chocolate Factory', 'Roald Dahl', 4],
    ['Number the Stars', 'Lois Lowry', 4]
]

@app.route('/', methods = ['POST', 'GET'])
def Home():
    return render_template('index.html')

@app.route('/r', methods = ['POST', "GET"])
def Recommend():
    with open('C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\ML\\HackJPS\\userinput.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Name', 'Author', 'Language Level', 'Fit'])
        i = 0
        for x in request.form.values():
            x = int(x)
            if x == 1 or x == 0:
                books[i].append(x)
                writer.writerow(books[i])
                i += 1
            else:
                books.pop(i)
                print('Hello')
                continue
        
        listbooks = []    
        for u in range(50):
            t = random.randint(300)
            o = l[t]
            y = [o[2]]
            e = model.predict([y])
            if e == 1:
                listbooks.append([o[0], o[1]])
                
    
    return render_template('index.html', recommendation=listbooks)
    
    
if __name__ == '__main__':
    app.run(debug=False)