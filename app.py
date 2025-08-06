import io
import pandas as pd
import joblib
import numpy as np
import streamlit as st

# --- CACHED FUNCTION TO LOAD MODEL ---
@st.cache_data
def load_model_bundle(model_path='cakingmodel.joblib'):
    """Loads the model bundle and caches it for performance."""
    model_bundle = joblib.load(model_path)
    return model_bundle


# --- HELPER FUNCTIONS ---    
def predict_caking_with_uncertainty(new_data_df, model_path="cakingmodel.joblib"):
    pipeline = joblib.load(model_path)['pipeline']
    scaler = pipeline.named_steps["scaler"]
    rf_model = pipeline.named_steps["rf"]

    new_data_scaled = scaler.transform(new_data_df)

    tree_predictions = [tree.predict(new_data_scaled) for tree in rf_model.estimators_]
    predictions_array = np.stack(tree_predictions)
    std_dev = np.std(predictions_array, axis=0)
    final_prediction = pipeline.predict(new_data_df)

    return final_prediction[0], std_dev[0]

def check_input_ranges(input_df, ranges):
    """Checks if inputs are within valid ranges and returns warning messages."""
    warnings = []
    is_in_range = True
    for feature in ranges.keys():
        min_val, max_val = ranges[feature]
        input_val = input_df[feature].iloc[0]
        if not (min_val <= input_val <= max_val):
            is_in_range = False
            warnings.append(f"**{feature}:** Input `{input_val}` is outside the recommended range of `{min_val}` to `{max_val}`.")
    return is_in_range, warnings


def batch_predict_with_uncertainty(df, model_path="cakingmodel.joblib"):
    """
    Loads a saved pipeline and makes predictions for an entire DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with multiple rows of input data.
    - model_path (str): Path to the saved model file.

    Returns:
    - tuple: A tuple containing two pandas Series (predictions, uncertainties).
    """
    pipeline = joblib.load(model_path)['pipeline']

    # --- Robustly get required model columns ---
    model_columns = list(pipeline_check.named_steps['scaler'].get_feature_names_out())

    # Ensure the dataframe has the required columns
    # Reorder df columns to match the model's training order
    df_for_prediction = df[model_columns]

    scaler = pipeline.named_steps["scaler"]
    rf_model = pipeline.named_steps["rf"]

    # Scale all data at once
    data_scaled = scaler.transform(df_for_prediction)

    # Get predictions from each tree for all rows at once
    tree_predictions = [tree.predict(data_scaled) for tree in rf_model.estimators_]

    # Stack predictions and calculate std dev and mean across all trees
    predictions_array = np.stack(tree_predictions)
    uncertainties = np.std(predictions_array, axis=0)
    predictions = np.mean(
        predictions_array, axis=0
    )  # This is what pipeline.predict() does

    return pd.Series(predictions), pd.Series(uncertainties)
    
def classify_caking_propensity(prediction, uncertainty):
    """
    Classifies caking propensity based on the upper bound of the prediction.
    """
    upper_bound = prediction + uncertainty
    if upper_bound <= 30:
        return "Low Caking Propensity", "success"
    elif 30 < upper_bound <= 60:
        return "Medium Caking Propensity", "warning"
    else:
        return "High Caking Propensity", "error"

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(
    page_title="Caking (%) Prediction",
    page_icon="🧪",
    layout="wide",
)
# Load the model bundle once using the cached function
model_bundle = load_model_bundle()
pipeline = model_bundle['pipeline']
feature_ranges = model_bundle['ranges']

st.title("Caking Propensity Prediction using Proximate Data")

st.warning(
    """
    **Disclaimer:** This is an exploratory tool and was trained on a limited dataset. 
    The predictions are estimates, not guarantees. Please use the provided uncertainty value 
    to gauge confidence and always supplement these results with professional judgment.
    """
)
st.write(
    "Choose your prediction method: Single or Multiple Samples"
)

# Display the recommended ranges clearly and permanently
range_text = "**Recommended Input Ranges (for a trusted prediction):**\n"
for feature, (min_val, max_val) in feature_ranges.items():
    range_text += f"- **{feature}:** `{np.round(min_val, 2)}` to `{np.round(max_val, 2)}`\n"
st.info(range_text)

# --- TABS FOR DIFFERENT MODES ---
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# --- TAB 1: SINGLE PREDICTION (INPUT BOXES) ---
with tab1:
    st.header("Predict from Manual Input")

    with st.form("single_prediction_form"):
        # Use columns for a cleaner layout
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Input Proximate Data")
            # --- MODIFICATION: Replaced st.slider with st.number_input ---
            inherent_moisture = st.number_input(
                label="Inherent Moisture (%)",
                min_value=0.0,
                max_value=100.0,
                value=None,
                step=0.1,
                format="%.1f",
            )
            ash = st.number_input(
                label="Ash (%)",
                min_value=0.0,
                max_value=100.0,
                value=None,
                step=0.1,
                format="%.1f",
            )
            # ADD MORE INPUT BOXES HERE...

        with col2:
            st.write("#### ")
            # --- MODIFICATION: Replaced st.slider with st.number_input ---
            volatile_matter = st.number_input(
                label="Volatile Matter (%)",
                min_value=0.0,
                max_value=100.0,
                value=None,
                step=0.1,
                format="%.1f",
            )
            fixed_carbon = st.number_input(
                label="Fixed Carbon (%)",
                min_value=0.0,
                max_value=100.0,
                value=None,
                step=0.01,
                format="%.1f",
            )
            # ADD MORE INPUT BOXES HERE...
        
        

        # --- SUBMIT BUTTON FOR THE FORM ---
        submitted = st.form_submit_button("Predict Caking Propensity (%)")

    if submitted:
        all_inputs = [inherent_moisture, ash, volatile_matter, fixed_carbon]
        if any(v is None for v in all_inputs):
            st.error('Please fill in all input fields before making a prediction.')
        else:
            input_data = {
                "Inherent Moisture": inherent_moisture,
                "Ash": ash,
                "Volatile Matter": volatile_matter,
                "Fixed Carbon": fixed_carbon,
                # ADD THE REST OF YOUR FEATURES HERE TO MATCH THE DICTIONARY
            }
            
            input_df = pd.DataFrame([input_data])

            is_in_range, range_warnings = check_input_ranges(input_df, feature_ranges)
            st.error(range_warnings)
            
            prediction, uncertainty = predict_caking_with_uncertainty(input_df)
    
            if prediction is not None:
                st.subheader("Prediction Result")
                st.metric(
                    label="Predicted Caking Propensity",
                    value=f"{prediction:.1f} %",
                    delta=f"± {uncertainty:.1f} % (Uncertainty)",
                    delta_color="off"
                )
                # Add a visual gauge/progress bar
                
    
                # Get the classification based on the new, simplified logic
                propensity, color = classify_caking_propensity(prediction, uncertainty)
                
                # Display the result in a colored box
                if color == "success":
                    st.success(f"**Classification: {propensity}**")
                elif color == "warning":
                    st.warning(f"**Classification: {propensity}**")
                else: # color == "error"
                    st.error(f"**Classification: {propensity}**")


# --- TAB 2: BATCH PREDICTION (FILE UPLOAD) ---
with tab2:
    # (This section remains unchanged)
    st.header("Predict from an Excel File")

    st.info(
    """
    Upload an Excel file (.xlsx) with columns matching the model's required inputs.
    
    **Required columns (in this order):**
    - Inherent Moisture
    - Ash
    - Volatile Matter
    - Fixed Carbon
    
    The app will process the file and allow you to download a new version containing the prediction and uncertainty columns.
    """
)

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        input_df_batch = pd.read_excel(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(input_df_batch.head())

        pipeline_check = joblib.load("cakingmodel.joblib")
        model_columns = list(pipeline_check.named_steps['scaler'].get_feature_names_out())

        missing_cols = set(model_columns) - set(input_df_batch.columns)
        if missing_cols:
            st.error(
                f"Error: Your file is missing the following required columns: {list(missing_cols)}"
            )
        else:
            st.success("File columns match model requirements. Ready to predict.")

            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing..."):
                    predictions, uncertainties = batch_predict_with_uncertainty(
                        input_df_batch
                    )

                    results_df = input_df_batch.copy()
                    results_df["Predicted_Caking (%)"] = predictions
                    results_df["Uncertainty (±%)"] = uncertainties

                    st.write("Prediction Results Preview:")
                    st.dataframe(results_df.head())

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        results_df.to_excel(
                            writer, index=False, sheet_name="Predictions"
                        )

                    processed_data = output.getvalue()

                    st.download_button(
                        label="📥 Download Results as Excel",
                        data=processed_data,
                        file_name="caking_prediction_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

# --- FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        Developed by <strong>SASOL Research & Technology: Feedstock</strong><br>
        © 2025 SASOL
    </div>
    """,
    unsafe_allow_html=True
)





