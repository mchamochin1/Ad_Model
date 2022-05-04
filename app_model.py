from flask import Flask, jsonify, request
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


#os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
#app.config['DEBUG'] = True


#
# CREO UNA PUERTA DE ACCESO A NUESTRO MODELO
# FULLSTACK CREARIA UN FRONTEND PARA ESTO
#
# Lo puedo interogar asi, para que me diga el impacto de invertir 
# en tv, radio y newspaper, cual sera el ROI.
#
# Se le interrogaria al modelo por ejemplo asi pasandole los parametros desde la Web
# http://127.0.0.1:5000/api/v1/predict?tv=10&radio=10&newspaper=10
# Con postmam podria meterle los parametros a mano y definit como se atacaria con 
# es url

# A.- Creo un ruta que me devuelva la prediccion del modelo
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('ad_model.pkl','rb')) # Leo el modelo con pickle
    
    # Cojete estos tres items y si no esta pon None
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict" # No eisten argumentos
    else:
        prediction = model.predict([[tv,radio,newspaper]]) # Predigo el modelo si me pasan las vars
    
    return jsonify({'predictions': prediction[0]})

# B.- Reentrenar de nuevo el modelo con los posibles nuevos registros que se recojan
@app.route('/api/v1/retrain', methods=['PUT'])
def retrain():
    
    os.chdir(os.path.dirname(__file__))

    data = pd.read_csv('data/Advertising.csv', index_col=0)

    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                    data['sales'],
                                                    test_size = 0.20,
                                                    random_state=42)

    model = Lasso(alpha=6000)
    model.fit(X_train, y_train)

    pickle.dump(model, open('ad_model.pkl', 'wb'))

    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    return "MSE es " + str(mse) + " RMSE es " + str(rmse)

app.run()
