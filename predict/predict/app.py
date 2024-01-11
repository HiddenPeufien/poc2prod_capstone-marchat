from flask import Flask, request, render_template
import json
from run import TextPredictionModel


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('render.html')
 
@app.route("/", methods=['POST'])
def get_prediction():
    model = TextPredictionModel.from_artefacts("train/data/artefacts/train/2023-01-04-17-25-24") #this did not work on my computer so i put the absolute path,
    #you might have to do the same i don't know why
    #model = TextPredictionModel.from_artefacts(
    #   "C:/Users/damie/OneDrive/Bureau/Nouveau dossier (9)/poc2prod_capstone-main/train/data/artefacts/train/2023-01-04-17-25-24")
    text = request.form['text']
    predictions = model.predict([text],top_k=1)
    
    return str(predictions)

@app.route("/get_prediction",methods=["GET"])
def request_prediction():
    model = TextPredictionModel.from_artefacts("train/data/artefacts/train/2023-01-04-17-25-24")
    text = request.args.get('text')
    predictions = model.predict([text],top_k=1)

    return str(predictions)

@app.route('/predict', methods=['POST'])
def request_post():
    body=json.loads(request.get_data())
    text_list = body['text']
    top_k = body['top_k']
    textPredictionModel = TextPredictionModel.from_artefacts('train/data/artefacts/train/2023-01-04-17-25-24')
    label_list = textPredictionModel.predict(text_list, top_k=top_k)
    print(label_list)
    return str(label_list)


if __name__ == '__main__':
    app.run()