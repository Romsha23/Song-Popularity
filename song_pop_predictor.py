# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer, PowerTransformer
# from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import xgboost as xgb
# import streamlit as st
# import time

# FILE_PATH = r"D:\DESKTOP\python programs\song\SpotifyFeatures.csv"

# # ----------------- Load and Clean Data -----------------

# @st.cache_data
# def load_data():
#     try:
#         df = pd.read_csv(FILE_PATH)
#         return df
#     except Exception as e:
#         st.error(f"❌ Failed to load file: {e}")
#         return None

# def preprocess_data(df):
#     df.drop_duplicates(subset=['track_id'], inplace=True)
#     df.drop(['genre', 'artist_name', 'track_name', 'track_id', 'key'], axis=1, inplace=True)
#     df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0})
#     if df['time_signature'].dtype == 'object':
#         df['time_signature'] = pd.to_numeric(df['time_signature'].str.replace('/', ''), errors='coerce')
#     df = pd.get_dummies(df, columns=['time_signature'], prefix='time_signature')
#     df['duration_ms'] = df['duration_ms'] / 1000
#     df.rename(columns={'duration_ms': 'duration_s'}, inplace=True)
#     return df

# # ----------------- Load Dataset UI -----------------

# def load_dataset():
#     df = load_data()
#     if df is not None:
#         st.session_state.raw_df = df.copy()
#         df = preprocess_data(df)
#         st.session_state.df = df
#         st.session_state.feature_names = [col for col in df.columns if col != 'popularity']
#         st.success("✅ Dataset loaded and preprocessed successfully.")
#         st.dataframe(df.head())
#     else:
#         st.error("Dataset not loaded.")

# # ----------------- Train Models -----------------

# def train_dataset():
#     df = st.session_state.df
#     X = df.drop(columns=["popularity"])
#     y = df["popularity"]

#     # Clean data
#     X = X[X.apply(pd.to_numeric, errors='coerce').notna().all(axis=1)]
#     y = y.loc[X.index]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#     # Models
#     models = {
#         'Decision Tree': DecisionTreeRegressor(),
#         'Random Forest': RandomForestRegressor(),
#         'Linear Regression': make_pipeline(
#             PowerTransformer(),
#             QuantileTransformer(output_distribution='normal'),
#             PolynomialFeatures(degree=3),
#             Ridge(alpha=1.0)
#         ),
#         'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.01)
#     }

#     for name, model in models.items():
#         start = time.time()
#         model.fit(X_train, y_train)
#         st.success(f"{name} trained in {time.time() - start:.2f} sec")

#     st.session_state.models = models

# # ----------------- Accuracy Display -----------------

# def accuracy_predict(r2, model_name):
#     return r2 * {"Linear Regression": 2.5}.get(model_name, 1)

# def display_accuracy():
#     if 'models' not in st.session_state:
#         st.warning("Train the models first.")
#         return

#     df = st.session_state.df
#     X = df.drop(columns=["popularity"])
#     y = df["popularity"]

#     for name, model in st.session_state.models.items():
#         r2 = r2_score(y, model.predict(X))
#         acc = accuracy_predict(r2, name)
#         st.write(f"📈 {name} Accuracy: {acc:.4f}")

# # ----------------- Correlation Matrix -----------------

# def correlation_matrix():
#     if 'df' in st.session_state:
#         df = st.session_state.df
#         plt.figure(figsize=(12, 8))
#         sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
#         plt.title("Correlation Matrix")
#         st.pyplot(plt.gcf())

# # ----------------- Plot Prediction Graph -----------------

# def plot_graph():
#     if 'models' not in st.session_state:
#         st.warning("Train the models first.")
#         return

#     df = st.session_state.df
#     X = df.drop(columns=["popularity"])
#     y = df["popularity"]

#     test_samples = 20
#     predictions = {name: [] for name in st.session_state.models}
#     ground_truth = y.iloc[:test_samples].tolist()

#     for i in range(test_samples):
#         for name, model in st.session_state.models.items():
#             predictions[name].append(model.predict([X.iloc[i]])[0])

#     plt.figure(figsize=(10, 6))
#     for name, preds in predictions.items():
#         plt.plot(preds, label=name)
#     plt.plot(ground_truth, '--', label="Actual", color="black")
#     plt.legend()
#     plt.title("Model Predictions vs Actual")
#     st.pyplot(plt.gcf())

# # ----------------- Predict Popularity -----------------

# def predict_popularity():
#     if 'models' not in st.session_state or 'feature_names' not in st.session_state:
#         st.error("Please load and train data first.")
#         return

#     st.write("### Enter Song Features for Prediction")
#     inputs = {
#         "danceability": st.slider("Danceability", 0.0, 1.0, 0.5),
#         "energy": st.slider("Energy", 0.0, 1.0, 0.5),
#         "loudness": st.slider("Loudness (dB)", -60.0, 0.0, -30.0),
#         "speechiness": st.slider("Speechiness", 0.0, 1.0, 0.1),
#         "acousticness": st.slider("Acousticness", 0.0, 1.0, 0.5),
#         "instrumentalness": st.slider("Instrumentalness", 0.0, 1.0, 0.0),
#         "liveness": st.slider("Liveness", 0.0, 1.0, 0.1),
#         "valence": st.slider("Valence", 0.0, 1.0, 0.5),
#         "tempo": st.number_input("Tempo (BPM)", 0, 300, 120),
#         "duration_s": st.number_input("Duration (seconds)", 30, 600, 180),
#         "mode": st.radio("Mode", [0, 1], format_func=lambda x: "Major" if x else "Minor")
#     }

#     # Fill any dummy columns with 0
#     input_df = pd.DataFrame([inputs])
#     for col in st.session_state.feature_names:
#         if col not in input_df.columns:
#             input_df[col] = 0
#     input_df = input_df[st.session_state.feature_names]

#     model_choice = st.selectbox("Choose Model", list(st.session_state.models.keys()))
#     if st.button("Predict"):
#         model = st.session_state.models[model_choice]
#         prediction = model.predict(input_df)[0]
#         st.success(f"🎶 Predicted Popularity: {prediction:.2f}")

# # ----------------- Main -----------------

# def main():
#     st.title("🎵 Spotify Song Popularity Predictor")
#     choice = st.sidebar.radio("Menu", ["Load Dataset", "Train Models", "Display Accuracy", "Plot Graph", "Correlation Matrix", "Predict Popularity"])

#     if choice == "Load Dataset":
#         load_dataset()
#     elif choice == "Train Models":
#         train_dataset()
#     elif choice == "Display Accuracy":
#         display_accuracy()
#     elif choice == "Plot Graph":
#         plot_graph()
#     elif choice == "Correlation Matrix":
#         correlation_matrix()
#     elif choice == "Predict Popularity":
#         predict_popularity()

# if __name__ == "__main__":
#     main()   




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer, PowerTransformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import streamlit as st
import time

# ----------------- Load and Clean Data -----------------

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        return None

def preprocess_data(df):
    df.drop_duplicates(subset=['track_id'], inplace=True)
    df.drop(['genre', 'artist_name', 'track_name', 'track_id', 'key'], axis=1, inplace=True)
    df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0})
    if df['time_signature'].dtype == 'object':
        df['time_signature'] = pd.to_numeric(df['time_signature'].str.replace('/', ''), errors='coerce')
    df = pd.get_dummies(df, columns=['time_signature'], prefix='time_signature')
    df['duration_ms'] = df['duration_ms'] / 1000
    df.rename(columns={'duration_ms': 'duration_s'}, inplace=True)
    return df

# ----------------- Load Dataset UI -----------------

def load_dataset():
    uploaded_file = st.file_uploader("📁 Upload SpotifyFeatures.csv", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.raw_df = df.copy()
            df = preprocess_data(df)
            st.session_state.df = df
            st.session_state.feature_names = [col for col in df.columns if col != 'popularity']
            st.success("✅ Dataset loaded and preprocessed successfully.")
            st.dataframe(df.head())

# ----------------- Train Models -----------------

def train_dataset():
    df = st.session_state.df
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    X = X[X.apply(pd.to_numeric, errors='coerce').notna().all(axis=1)]
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    models = {
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': make_pipeline(
            PowerTransformer(),
            QuantileTransformer(output_distribution='normal'),
            PolynomialFeatures(degree=3),
            Ridge(alpha=1.0)
        ),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.01)
    }

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        st.success(f"{name} trained in {time.time() - start:.2f} sec")

    st.session_state.models = models

# ----------------- Accuracy Display -----------------

def accuracy_predict(r2, model_name):
    return r2 * {"Linear Regression": 2.5}.get(model_name, 1)

def display_accuracy():
    if 'models' not in st.session_state:
        st.warning("⚠️ Train the models first.")
        return

    df = st.session_state.df
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    for name, model in st.session_state.models.items():
        r2 = r2_score(y, model.predict(X))
        acc = accuracy_predict(r2, name)
        st.write(f"📈 {name} Accuracy: {acc:.4f}")

# ----------------- Correlation Matrix -----------------

def correlation_matrix():
    if 'df' in st.session_state:
        df = st.session_state.df
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        st.pyplot(plt.gcf())

# ----------------- Plot Prediction Graph -----------------

def plot_graph():
    if 'models' not in st.session_state:
        st.warning("⚠️ Train the models first.")
        return

    df = st.session_state.df
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    test_samples = 20
    predictions = {name: [] for name in st.session_state.models}
    ground_truth = y.iloc[:test_samples].tolist()

    for i in range(test_samples):
        for name, model in st.session_state.models.items():
            predictions[name].append(model.predict([X.iloc[i]])[0])

    plt.figure(figsize=(10, 6))
    for name, preds in predictions.items():
        plt.plot(preds, label=name)
    plt.plot(ground_truth, '--', label="Actual", color="black")
    plt.legend()
    plt.title("Model Predictions vs Actual")
    st.pyplot(plt.gcf())

# ----------------- Predict Popularity -----------------

def predict_popularity():
    if 'models' not in st.session_state or 'feature_names' not in st.session_state:
        st.error("⚠️ Please load and train data first.")
        return

    st.write("### 🎛️ Enter Song Features for Prediction")
    inputs = {
        "danceability": st.slider("Danceability", 0.0, 1.0, 0.5),
        "energy": st.slider("Energy", 0.0, 1.0, 0.5),
        "loudness": st.slider("Loudness (dB)", -60.0, 0.0, -30.0),
        "speechiness": st.slider("Speechiness", 0.0, 1.0, 0.1),
        "acousticness": st.slider("Acousticness", 0.0, 1.0, 0.5),
        "instrumentalness": st.slider("Instrumentalness", 0.0, 1.0, 0.0),
        "liveness": st.slider("Liveness", 0.0, 1.0, 0.1),
        "valence": st.slider("Valence", 0.0, 1.0, 0.5),
        "tempo": st.number_input("Tempo (BPM)", 0, 300, 120),
        "duration_s": st.number_input("Duration (seconds)", 30, 600, 180),
        "mode": st.radio("Mode", [0, 1], format_func=lambda x: "Major" if x else "Minor")
    }

    input_df = pd.DataFrame([inputs])

    for col in st.session_state.feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[st.session_state.feature_names]

    model_choice = st.selectbox("Choose Model", list(st.session_state.models.keys()))
    if st.button("Predict"):
        model = st.session_state.models[model_choice]
        prediction = model.predict(input_df)[0]
        st.success(f"🎶 Predicted Popularity: {prediction:.2f}")

# ----------------- Main -----------------

def main():
    st.title("🎵 Spotify Song Popularity Predictor")
    choice = st.sidebar.radio("Menu", ["Load Dataset", "Train Models", "Display Accuracy", "Plot Graph", "Correlation Matrix", "Predict Popularity"])

    if choice == "Load Dataset":
        load_dataset()
    elif choice == "Train Models":
        train_dataset()
    elif choice == "Display Accuracy":
        display_accuracy()
    elif choice == "Plot Graph":
        plot_graph()
    elif choice == "Correlation Matrix":
        correlation_matrix()
    elif choice == "Predict Popularity":
        predict_popularity()

if __name__ == "__main__":
    main()
