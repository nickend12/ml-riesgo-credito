import os
import pandas as pd
from datasets import load_dataset


def cargar_datos_huggingface(dataset_id: str = "imodels/credit-card") -> pd.DataFrame:
    """
    Descarga automáticamente el dataset desde Hugging Face
    y lo convierte a un DataFrame de Pandas.
    """
    print(f"📡 Descargando dataset desde Hugging Face: '{dataset_id}'...")
    
    # Leer el token de las variables de entorno si está disponible
    hf_token = os.getenv("HF_TOKEN")
    
    # Carga del conjunto de datos desde Hugging Face
    dataset = load_dataset(dataset_id, token=hf_token)
    
    # Selección de la partición principal (train)
    if 'train' in dataset:
        df = pd.DataFrame(dataset['train'])
    else:
        split_nombre = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[split_nombre])
        
    print("✅ Ingesta finalizada con éxito.")
    print(f"📊 Tamaño del DataFrame: {df.shape[0]} filas x {df.shape[1]} columnas.\n")
    return df


if __name__ == "__main__":
    df = cargar_datos_huggingface()
    print("Muestra de los primeros 3 registros:")
    print(df.head(3))