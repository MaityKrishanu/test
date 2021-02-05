import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json 
import pickle
vocab_size = 12708
maxlen = 30
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    temp=request.get_data(as_text=True)
    new=[temp]
    tok = tokenizer1.texts_to_sequences(new)
    pad = pad_sequences(tok, padding='post', maxlen=maxlen)
    y_pred1 = loaded_model.predict(pad)
    pred = np.argmax(y_pred1, axis=-1)
    if pred == 1:
        output = "Bully"
    else:
        output = "Non_Bully"

    

    return render_template('index.html', prediction_text='Input sentence indicates : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)