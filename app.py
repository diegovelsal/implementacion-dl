from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from gensim.models import FastText

app = Flask(__name__)

# Carga tu modelo
model = tf.keras.models.load_model('rnn_cuisine_classifier_v3.h5')

# Define las clases (etiquetas de cocinas)
classes = ["italian", "mexican", "southern_us", "indian", "chinese", "french", "thai","cajun_creole", "japanese", "greek", "spanish", "vietnamese", "korean", "moroccan", "british", "filipino", "irish", "jamaican", "russian", "brazilian"]
  # Cambia según tu modelo

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get ingredients from the textarea
    ingredients = request.form.get('ingredients')  # Input as a single string with line breaks

    # Split the ingredients into a list based on line breaks
    ingredient_list = ingredients.split('\n')

    # Clean the list (remove empty strings, strip whitespace)
    ingredient_list = [ingredient.strip() for ingredient in ingredient_list if ingredient.strip()]

    # Preprocess the list for your model
    ft_model = FastText.load("models/ft_100_3_2_1_10.model")
    preprocessed = preprocess_ingredients(ingredient_list, ft_model, 14)

    # Agregar una dimensión para que coincida con la entrada del modelo (batch_size, max_len, vector_size)
    preprocessed = np.expand_dims(preprocessed, axis=0)  # Shape: (1, max_len, vector_size)
    
    # Hacer predicción con el modelo
    predictions = model.predict(preprocessed)  # Salida: probabilidad para cada clase
    
    # Obtener el índice de la clase con la mayor probabilidad
    predicted_label_index = np.argmax(predictions, axis=1)[0]  # Índice de la clase
    
    # Mapear el índice a la etiqueta de la clase
    predicted_class = classes[predicted_label_index]  # La clase predicha
    
    # Retornar el resultado como JSON
    return render_template('index.html', prediction=f"Your dish seems to be from {predicted_class.capitalize()} cuisine!")


def preprocess_ingredients(ingredient_list, ft_model, max_len=10):
    """
    Convierte una lista de ingredientes en una secuencia de embeddings.
    Rellena con vectores de ceros si la lista tiene menos de `max_len` ingredientes.

    Parámetros:
    - ingredient_list: lista de ingredientes como strings.
    - model: modelo FastText entrenado.
    - max_len: longitud máxima de la secuencia de embeddings.

    Retorna:
    - Un array numpy de shape (max_len, vector_size) listo para el modelo.
    """
    # Obtener embeddings de cada ingrediente si existe en el modelo
    embeddings = [ft_model.wv[ingredient] for ingredient in ingredient_list if ingredient in ft_model.wv]
    
    # Recortar o completar con vectores de ceros según `max_len`
    if len(embeddings) < max_len:
        embeddings.extend([np.zeros(ft_model.vector_size)] * (max_len - len(embeddings)))
    else:
        embeddings = embeddings[:max_len]
    
    return np.array(embeddings)

if __name__ == "__main__":
    app.run(debug=True)
