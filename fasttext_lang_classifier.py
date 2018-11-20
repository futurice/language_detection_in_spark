# We need fasttext to load the model and make predictions
import fasttext

# Load model (loads when this library is being imported)
model = fasttext.load_model('data/model_fasttext.bin')

# This is the function we use in UDF to predict the language of a given msg
def predict_language(msg):
    pred = model.predict([msg])[0][0]
    pred = pred.replace('__label__', '')
    return pred