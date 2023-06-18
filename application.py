from flask import Flask,request,render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

#root
@app.route(rule='/')
def index():
    return render_template('index.html') 

@app.route(rule='/predictdata',methods=['GET','POST']) # type: ignore
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        
        data = CustomData(
            gender=str(request.form.get('gender')),
            race_ethnicity=str(request.form.get('ethnicity')),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=str(request.form.get('lunch')),
            test_preparation_course=str(request.form.get("test_preparation_course")),
            reading_score=float(str(request.form.get("reading_score"))),
            writing_score=float(str(request.form.get("writing_score")))
        )
        df = data.get_data_as_data_frame()
        print(df)

        results = PredictPipeline().predict(df)

        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True) # at deployment phase remove debug