from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import h5py
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load model ResNet50 tanpa top layer
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Load fitur dan nama file dari HDF5
with h5py.File('model/features.h5', 'r') as h5f:
    features = np.array(h5f['features'])
    image_names = [name.decode('utf-8') for name in h5f['image_names']]

# Load narasi
narrative_df = pd.read_csv('narasi-karmawibhangga.csv')
narrative_dict = {
    row['filename']: {
        'tema': row['Tema'],
        'narasi': row['Narasi'],
        'makna_moral': row['Makna moral']
    } for _, row in narrative_df.iterrows()
}

# Fungsi preprocessing query image
def preprocess_query_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Fungsi cari gambar serupa
def find_similar(query_feature, features, top_k=3):
    sims = cosine_similarity(query_feature, features)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return top_indices, sims[top_indices]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filepath = os.path.join('static', 'query.jpg')
    file.save(filepath)

    # Ekstraksi fitur query
    query_img = preprocess_query_image(filepath)
    query_feature = model.predict(query_img)

    indices, scores = find_similar(query_feature, features, top_k=1)

    idx = indices[0]
    score = scores[0]
    fname = image_names[idx]
    narasi = narrative_dict.get(fname, {
        'tema': 'Tidak diketahui',
        'narasi': 'Tidak ditemukan',
        'makna_moral': 'Tidak tersedia'
    })

    result = {
        'filename': fname,
        'similarity': float(score),
        **narasi
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)