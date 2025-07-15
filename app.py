import json
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Cargar dominios BIAN desde CSV
bian_df = pd.read_csv("bian_domains.csv")
print("📊 Dominios BIAN cargados:", bian_df.shape[0], "dominios.")

# Cargar modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Archivo para guardar los embeddings
embeddings_file = "bian_embeddings.npy"

# Cargar embeddings si existen, sino generarlos y guardarlos
if os.path.exists(embeddings_file):
    print("📁 Cargando embeddings existentes...")
    bian_embeddings = np.load(embeddings_file)
else:
    print("🔄 Generando embeddings de dominios BIAN...")
    bian_embeddings = model.encode(bian_df['description'].tolist(), convert_to_numpy=True)
    np.save(embeddings_file, bian_embeddings)
    print(f"💾 Embeddings guardados en {embeddings_file}")

# Cargar servicios a evaluar
with open('myservices.json') as f:
    services = json.load(f)

for service in services:
    service_embedding = model.encode(service['description'], convert_to_numpy=True)

    # Calcular similitudes coseno (1 - distancia coseno)
    similarities = [1 - cosine(service_embedding, emb) for emb in bian_embeddings]

    # Mejor dominio y su confianza
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    matched_domain = bian_df.iloc[best_idx]['domain_name']

    print(f"🔹 Servicio: {service['name']}")
    print(f"↪️  Sugerido: {matched_domain} (confianza: {best_score:.2f})\n")
    """ proceso para generar BOM """
    
