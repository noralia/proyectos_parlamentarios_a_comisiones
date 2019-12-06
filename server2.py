from flask import Flask,  jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)
parser = reqparse.RequestParser()
parser.add_argument("texto")

@app.route('/classify', methods=['POST'])
def classify():

     args= parser.parse_args()
     texto =  request.form['texto']
     textos = []
     textos.append(texto)
     # clasificar los proyectos de test
     labels_considerados, puntajes = tc.classify(
          classifier_name="giros_classifier",
          examples=textos
     )
     code = sorted(zip(puntajes[0], labels_considerados), reverse=True)[0][1]
     mask = [bool(int(char)) for char in code]
     lista = comisiones[mask].index.values
     set_comisiones1 = set()
     set_comisiones2 = set()
     set_comisiones1 |= set(lista)
     for j in range(1, 2):
          code2 = sorted(zip(puntajes[0], labels_considerados), reverse=True)[j][1]
          mask2 = [bool(int(char)) for char in code2]
          lista2 = comisiones[mask2].index.values
          set_comisiones2 |= set(lista2)
     ids, distancias, palabras_comunes = tc.get_similar(
          example=texto,
          max_similars=8
     )
     indices = [int(x) for x in ids]
     titulos = proy.iloc[indices]['TITULO']
     dict = sorted(zip(distancias, palabras_comunes))

     return jsonify(textos,
                    lista.tolist(),
                    [item for item in set_comisiones2],
                    dict
                    )

@app.route('/compare', methods=['POST'])
def predict():
     texto = request.json
     textos = texto
     print(texto)
     ids, distancias, palabras_comunes = tc.get_similar(
          example='MODIFICACION DE LA LEY 20744 DE CONTRATO DE TRABAJO',
          max_similars=8
     )
     indices = [int(x) for x in ids]
     print(indices)
     print(proy.iloc[indices]['TITULO'])
     dict = sorted(zip(distancias, palabras_comunes))
     print(palabras_comunes)
     print(comisiones)
     return jsonify(dict)

if __name__ == '__main__':
     tc = joblib.load('../data/giros_classifier.sav')
     comisiones = []
     comisiones = joblib.load('../data/lista_.sav')
     proy = pd.read_csv("../data/proy_ME_2.csv", encoding="utf-8")
     app.run(port=8080)