import numpy as np
from flask import Flask,request,jsonify,render_template
import joblib
import sqlite3
import pandas as pd
#import cv2



app = Flask(__name__)


dataset = pd.read_csv('processed.csv')
X = dataset[['IPC1', 'IPC2', 'IPC3', 'IAC1', 'IAC2', 'LLC1', 'LLC2', 'ISK1', 'ISK3',
       'IAS1', 'IT2', 'IT3', 'IB1', 'IE1', 'IE2', 'ILR1']]
Y = dataset['label']

from sklearn.ensemble import RandomForestClassifier
#now train Random Forest algorithm
rf_cls = RandomForestClassifier(n_estimators=150,criterion='entropy')
rf_cls.fit(X, Y)
predict = rf_cls.predict(X)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")


@app.route('/index')
def index():
	return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():

    int_features= [float(x) for x in request.form.values()]
    print(int_features,len(int_features))
    final4=[np.array(int_features)]

    #final_features = np.array([val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14,val15,val16,val17,val18]).reshape(1,-1)
    #model = joblib.load('model1.sav')
    predict = rf_cls.predict(final4)
    if predict == 0:
        output = 'Student Learning Behaviour is Excellent!'
    elif predict == 1:
        output = 'Student Learning Behaviour is Medium!'
    elif predict == 2:
        output = 'Student Learning Behaviour is Poor!'
 
    return render_template('result.html',output=output)





@app.route('/notebook')
def notebook1():
	return render_template('StudentLearningBehaviour.html')



@app.route('/about')
def about():
	return render_template('about.html')

if __name__ == "__main__":
    app.run()
