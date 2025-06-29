import streamlit as st
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Judul Aplikasi
st.set_page_config(page_title="Prediksi Kualitas Wine", layout="wide")
st.title("🍷 Prediksi Kualitas Red Wine")
st.write("""
Aplikasi ini memprediksi kualitas wine (0-10) menggunakan Decision Tree.
Data source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
""")

# Fungsi Load Data dari Cloud
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    return pd.read_csv(url, sep=';')

# Fungsi Train Model
@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier().fit(X_train, y_train)
    return model

# Load Model
model = train_model()

# Sidebar untuk Input
with st.sidebar:
    st.header("Parameter Wine")
    fixed_acidity = st.slider('Fixed Acidity', 4.0, 16.0, 7.4)
    volatile_acidity = st.slider('Volatile Acidity', 0.1, 2.0, 0.7)
    alcohol = st.slider('Alcohol', 8.0, 15.0, 9.4)
    # Tambahkan 8 input lainnya...

# Tombol Prediksi
if st.button("Prediksi Kualitas"):
    input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, alcohol]], 
                            columns=['fixed acidity', 'volatile acidity', 'alcohol'])
    prediction = model.predict(input_data)[0]
    
    st.success(f'**Prediksi Kualitas Wine:** {prediction}/10')
    
    # Visualisasi
    st.bar_chart(load_data()['quality'].value_counts())

# Penjelasan Cloud Computing
with st.expander("ℹ️ Tentang Implementasi Cloud"):
    st.write("""
    **Cloud Computing dalam Proyek Ini:**
    - **Sumber Data**: Dataset di-load langsung dari UCI (cloud-based)
    - **Model Training**: Proses komputasi dilakukan lokal dengan caching via Streamlit
    - **Deployment**: Aplikasi dapat di-deploy ke Streamlit Cloud (PaaS)
    """)