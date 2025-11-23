# Stacking_Regression_app.py
import os
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ---- Model loading ----
current_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = "stacking_reg_diabetes.pkl"
model_path = os.path.join(current_dir, model_filename)

model = None
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.warning(f"Model file found but failed to load: {e}")
else:
    # Show friendly error on app if model missing
    st.error(
        f"Model file not found at `{model_path}`. "
        "Make sure `stacking_reg_diabetes.pkl` is in the same folder as this file and committed to your repo."
    )

def predict(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions using the loaded model and return a DataFrame with predictions and original features.
    """
    if model is None:
        raise RuntimeError("Model is not loaded.")
    preds = model.predict(data)
    prediction = pd.DataFrame(preds, columns=["target"])
    prediction = pd.concat([prediction, data.reset_index(drop=True)], axis=1)
    return prediction

def main():
    st.set_page_config(layout="wide")
    st.title("Stacking_Regression")
    st.sidebar.title("Stacking_Regression")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Stacking_Regression</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")

    uploadedFile = st.sidebar.file_uploader(
        "Choose a file (CSV or XLSX)", type=["csv", "xlsx"], accept_multiple_files=False
    )

    data = None
    if uploadedFile is not None:
        try:
            # try csv first
            uploadedFile.seek(0)
            data = pd.read_csv(uploadedFile)
        except Exception:
            try:
                uploadedFile.seek(0)
                data = pd.read_excel(uploadedFile)
            except Exception as e:
                st.sidebar.error("Unable to read the uploaded file. Make sure it's a valid CSV or Excel file.")
                st.sidebar.exception(e)

    st.sidebar.markdown("---")
    st.sidebar.write("Model file expected at:")
    st.sidebar.code(model_path)

    # Only enable Predict if model is loaded and data is available
    if st.button("Predict"):
        if model is None:
            st.error("Prediction not possible: the model is not loaded. Upload the model file to the repo.")
            return
        if data is None or data.shape[0] == 0:
            st.error("Please upload a valid CSV/XLSX file with rows to predict.")
            return

        # Run prediction and display
        try:
            result = predict(data)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return

        # Display results as table
        try:
            import seaborn as sns
            cm = sns.light_palette("blue", as_cmap=True)
            st.table(result.style.background_gradient(cmap=cm))
        except Exception:
            # fallback if seaborn not available
            st.dataframe(result)

if __name__ == "__main__":
    main()

