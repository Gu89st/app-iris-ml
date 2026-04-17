import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd # Importante para mostrar el histórico en una tabla

# Credenciales de la BD
USER = "postgres.wrxhksisfzqbgwpgypep"
PASSWORD = "databaseALT_99202"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# Función para obtener la conexión a la BD
def get_db_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

# Función para inicializar la tabla si no existe
def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iris_predictions (
                id SERIAL PRIMARY KEY,
                sepal_length NUMERIC,
                sepal_width NUMERIC,
                petal_length NUMERIC,
                petal_width NUMERIC,
                predicted_species VARCHAR(50),
                confidence NUMERIC,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.sidebar.error(f"Error inicializando BD: {e}")

# Ejecutar inicialización de la BD
init_db()

# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Estandarizar
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        confidence = float(max(probabilities))
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{confidence:.1%}**")
        
        # Guardar en la base de datos
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO iris_predictions 
                (sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence))
            conn.commit()
            cursor.close()
            conn.close()
            st.toast("✅ Predicción guardada en la base de datos")
        except Exception as e:
            st.error(f"Error al guardar en la base de datos: {e}")

    # --- SECCIÓN DEL HISTÓRICO ---
    st.divider()
    st.header("📊 Histórico de Predicciones")
    
    try:
        # Cargar los datos ordenados por fecha descendente usando Pandas
        conn = get_db_connection()
        query = "SELECT * FROM iris_predictions ORDER BY created_at DESC"
        df_history = pd.read_sql(query, conn)
        conn.close()
        
        if not df_history.empty:
            # Damos un formato más amigable a la columna de fecha y a la confianza
            df_history['created_at'] = pd.to_datetime(df_history['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df_history['confidence'] = (df_history['confidence'] * 100).apply(lambda x: f"{x:.1f}%")
            
            # Mostramos el dataframe en Streamlit
            st.dataframe(df_history, use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay predicciones en el historial.")
            
    except Exception as e:
        st.error(f"No se pudo cargar el historial: {e}")
