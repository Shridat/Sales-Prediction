from flask import Flask,render_template,request
import pickle
import numpy as np 
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
   return  render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    store = request.form['store']
    dept = request.form['dept']
    hol = request.form['holiday']
    cpi = request.form['cpi']
    quant = request.form['quant']
    size = request.form['size']
    week = request.form['week']
    type = request.form['type']
    year = request.form['year']
    #features = [store,dept,cpi,quant,week]
    #int_feature = [int(x) for x in features]
    features = [store,dept,hol,cpi,quant,size,week,type,year]
    final_features = [np.array(features)]
    #prediction = model.predict([[1,1,False,211,8,152345,5,1,2022]])
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('result.html',store=store,dept=dept,hol=hol,cpi=cpi,quant=quant,size=size,week=week,type=type,year=year,predicted_text='Expected weekly sales will be ₹ :{}'.format(str(output)))

app.run(debug=True)