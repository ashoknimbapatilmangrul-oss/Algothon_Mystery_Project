import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Mystery Dataset Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Mystery Dataset Analyzer")
st.write(
    "Upload any CSV dataset and automatically clean, analyze, "
    "visualize, detect anomalies, test relationships and build predictions."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🔍 Analysis Menu")

section = st.sidebar.radio(
    "Choose Section",
    [
        "1. Dataset Upload",
        "2. Data Quality",
        "3. Data Cleaning",
        "4. Variable Understanding",
        "5. Statistics",
        "6. Visualization",
        "7. Correlation Analysis",
        "8. Anomaly Detection",
        "9. Hypothesis Testing",
        "10. Machine Learning",
        "11. Automatic Insights",
        "12. Download Cleaned Data"
    ]
)


# =========================================================
# DATA UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload your CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info("👆 Please upload a CSV file to start analysis.")

    st.markdown("""
    ### How this project works

    **Upload CSV → Clean Data → Analyze → Visualize → Detect Anomalies
    → Test Relationships → Machine Learning → Prediction**

    ### Features

    - Data cleaning
    - Missing value detection
    - Duplicate detection
    - Variable understanding
    - Statistical analysis
    - Visualizations
    - Correlation analysis
    - Anomaly detection
    - Hypothesis testing
    - Machine learning
    - Automatic insights
    - Cleaned dataset download
    """)

    st.stop()


# =========================================================
# READ CSV
# =========================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read CSV file: {e}")
    st.stop()


if df.empty:

    st.error("The uploaded CSV file is empty.")
    st.stop()


# Keep original copy
original_df = df.copy()


# =========================================================
# COMMON VARIABLES
# =========================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()


# =========================================================
# 1. DATASET UPLOAD
# =========================================================

if section == "1. Dataset Upload":

    st.header("📂 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Numeric Columns", len(numeric_columns))

    with col4:
        st.metric("Categorical Columns", len(categorical_columns))

    st.subheader("First 10 Rows")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Shape")

    st.write(
        f"Rows = **{df.shape[0]}** | "
        f"Columns = **{df.shape[1]}**"
    )


# =========================================================
# 2. DATA QUALITY
# =========================================================

elif section == "2. Data Quality":

    st.header("🔎 Data Quality")

    missing_count = df.isnull().sum()
    duplicate_count = df.duplicated().sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Missing Values",
            int(missing_count.sum())
        )

    with col2:
        st.metric(
            "Duplicate Rows",
            int(duplicate_count)
        )

    with col3:
        st.metric(
            "Total Rows",
            len(df)
        )

    st.subheader("Missing Values by Column")

    quality_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_count.values,
        "Missing %": (
            missing_count.values / len(df) * 100
        ).round(2)
    })

    st.dataframe(
        quality_df,
        use_container_width=True
    )

    st.subheader("Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )


# =========================================================
# 3. DATA CLEANING
# =========================================================

elif section == "3. Data Cleaning":

    st.header("🧹 Data Cleaning")

    cleaned_df = df.copy()

    original_rows = len(cleaned_df)

    duplicate_rows = cleaned_df.duplicated().sum()

    if duplicate_rows > 0:

        cleaned_df = cleaned_df.drop_duplicates()

    # Fill missing numeric values with median
    for column in cleaned_df.select_dtypes(
        include=np.number
    ).columns:

        if cleaned_df[column].isnull().sum() > 0:

            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].median()
            )

    # Fill missing categorical values with mode
    for column in cleaned_df.select_dtypes(
        exclude=np.number
    ).columns:

        if cleaned_df[column].isnull().sum() > 0:

            mode_value = cleaned_df[column].mode()

            if len(mode_value) > 0:

                cleaned_df[column] = cleaned_df[column].fillna(
                    mode_value.iloc[0]
                )

    st.success("✅ Data cleaning completed.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Rows",
            original_rows
        )

    with col2:
        st.metric(
            "Duplicates Removed",
            int(duplicate_rows)
        )

    with col3:
        st.metric(
            "Final Rows",
            len(cleaned_df)
        )

    st.subheader("Cleaned Dataset")

    st.dataframe(
        cleaned_df,
        use_container_width=True
    )

    st.subheader("Remaining Missing Values")

    st.write(
        int(cleaned_df.isnull().sum().sum())
    )


# =========================================================
# 4. VARIABLE UNDERSTANDING
# =========================================================

elif section == "4. Variable Understanding":

    st.header("📋 Variable Understanding")

    variable_df = pd.DataFrame({
        "Variable": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Unique Values": [
            df[column].nunique()
            for column in df.columns
        ],
        "Missing Values": [
            df[column].isnull().sum()
            for column in df.columns
        ]
    })

    st.dataframe(
        variable_df,
        use_container_width=True
    )

    st.subheader("Numeric Variables")

    if numeric_columns:

        for column in numeric_columns:

            st.write(
                f"**{column}** → Numeric variable"
            )

    else:

        st.warning("No numeric variables found.")


    st.subheader("Categorical Variables")

    if categorical_columns:

        for column in categorical_columns:

            st.write(
                f"**{column}** → Categorical variable"
            )

    else:

        st.info("No categorical variables found.")


# =========================================================
# 5. STATISTICS
# =========================================================

elif section == "5. Statistics":

    st.header("📈 Statistical Analysis")

    if not numeric_columns:

        st.warning(
            "No numeric columns available for statistical analysis."
        )

    else:

        st.subheader("Descriptive Statistics")

        st.dataframe(
            df[numeric_columns].describe().T,
            use_container_width=True
        )

        selected_column = st.selectbox(
            "Select numeric variable",
            numeric_columns
        )

        column_data = df[selected_column].dropna()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Mean",
                f"{column_data.mean():.2f}"
            )

        with col2:
            st.metric(
                "Median",
                f"{column_data.median():.2f}"
            )

        with col3:
            st.metric(
                "Minimum",
                f"{column_data.min():.2f}"
            )

        with col4:
            st.metric(
                "Maximum",
                f"{column_data.max():.2f}"
            )


# =========================================================
# 6. VISUALIZATION
# =========================================================

elif section == "6. Visualization":

    st.header("📊 Data Visualization")

    if len(numeric_columns) < 1:

        st.warning(
            "At least one numeric column is required."
        )

    else:

        selected_column = st.selectbox(
            "Select variable",
            numeric_columns
        )

        st.subheader(
            f"Distribution of {selected_column}"
        )

        st.bar_chart(
            df[selected_column]
        )

        if len(numeric_columns) >= 2:

            st.subheader("Relationship Between Two Variables")

            x_column = st.selectbox(
                "X-axis",
                numeric_columns,
                key="x_column"
            )

            y_options = [
                col for col in numeric_columns
                if col != x_column
            ]

            if y_options:

                y_column = st.selectbox(
                    "Y-axis",
                    y_options,
                    key="y_column"
                )

                chart_df = df[
                    [x_column, y_column]
                ].dropna()

                st.scatter_chart(
                    chart_df,
                    x=x_column,
                    y=y_column
                )


# =========================================================
# 7. CORRELATION ANALYSIS
# =========================================================

elif section == "7. Correlation Analysis":

    st.header("🔗 Correlation Analysis")

    if len(numeric_columns) < 2:

        st.warning(
            "At least two numeric columns are required."
        )

    else:

        correlation = df[numeric_columns].corr()

        st.subheader("Correlation Matrix")

        st.dataframe(
            correlation.round(3),
            use_container_width=True
        )

        st.subheader("Strongest Relationships")

        pairs = []

        for i in range(len(numeric_columns)):

            for j in range(i + 1, len(numeric_columns)):

                col1 = numeric_columns[i]
                col2 = numeric_columns[j]

                value = correlation.loc[
                    col1,
                    col2
                ]

                if not pd.isna(value):

                    pairs.append(
                        (
                            col1,
                            col2,
                            value,
                            abs(value)
                        )
                    )

        pairs = sorted(
            pairs,
            key=lambda x: x[3],
            reverse=True
        )

        if pairs:

            for col1, col2, value, _ in pairs[:5]:

                st.write(
                    f"**{col1} ↔ {col2}** : "
                    f"Correlation = **{value:.3f}**"
                )


# =========================================================
# 8. ANOMALY DETECTION
# =========================================================

elif section == "8. Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    if not numeric_columns:

        st.warning(
            "No numeric columns available."
        )

    else:

        selected_column = st.selectbox(
            "Select variable for anomaly detection",
            numeric_columns
        )

        values = df[selected_column].dropna()

        if len(values) < 4:

            st.warning(
                "Not enough data for IQR anomaly detection."
            )

        else:

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)

            iqr = q3 - q1

            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            anomalies = df[
                (df[selected_column] < lower_limit)
                |
                (df[selected_column] > upper_limit)
            ]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Q1",
                    f"{q1:.2f}"
                )

            with col2:
                st.metric(
                    "Q3",
                    f"{q3:.2f}"
                )

            with col3:
                st.metric(
                    "Lower Limit",
                    f"{lower_limit:.2f}"
                )

            with col4:
                st.metric(
                    "Upper Limit",
                    f"{upper_limit:.2f}"
                )

            st.subheader("Detected Unusual Values")

            if len(anomalies) == 0:

                st.success(
                    "✅ No unusual values detected."
                )

            else:

                st.warning(
                    f"⚠️ {len(anomalies)} unusual value(s) detected."
                )

                st.dataframe(
                    anomalies,
                    use_container_width=True
                )


# =========================================================
# 9. HYPOTHESIS TESTING
# =========================================================

elif section == "9. Hypothesis Testing":

    st.header("🧪 Hypothesis Testing")

    if len(numeric_columns) < 2:

        st.warning(
            "At least two numeric variables are required."
        )

    else:

        x_column = st.selectbox(
            "Select Variable X",
            numeric_columns,
            key="hyp_x"
        )

        y_options = [
            col for col in numeric_columns
            if col != x_column
        ]

        y_column = st.selectbox(
            "Select Variable Y",
            y_options,
            key="hyp_y"
        )

        test_df = df[
            [x_column, y_column]
        ].dropna()

        if len(test_df) < 3:

            st.warning(
                "Not enough valid observations."
            )

        else:

            correlation_value, p_value = pearsonr(
                test_df[x_column],
                test_df[y_column]
            )

            st.subheader("Hypotheses")

            st.write(
                f"**H₀:** There is no significant relationship "
                f"between {x_column} and {y_column}."
            )

            st.write(
                f"**H₁:** There is a significant relationship "
                f"between {x_column} and {y_column}."
            )

            st.write("Significance level: **α = 0.05**")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Pearson Correlation",
                    f"{correlation_value:.4f}"
                )

            with col2:

                st.metric(
                    "p-value",
                    f"{p_value:.6f}"
                )

            if p_value < 0.05:

                st.success(
                    "Reject H₀ → Significant relationship detected."
                )

            else:

                st.info(
                    "Fail to reject H₀ → "
                    "No statistically significant relationship detected."
                )


# =========================================================
# 10. MACHINE LEARNING
# =========================================================

elif section == "10. Machine Learning":

    st.header("🤖 Machine Learning Prediction")

    if len(numeric_columns) < 2:

        st.warning(
            "At least two numeric columns are required."
        )

    else:

        st.write(
            "The model predicts one numeric variable using "
            "other numeric variables."
        )

        # Remove ID-like columns
        feature_candidates = []

        for column in numeric_columns:

            name = column.lower()

            unique_ratio = (
                df[column].nunique(dropna=True)
                / max(len(df), 1)
            )

            if (
                "id" not in name
                and "code" not in name
                and unique_ratio < 0.95
            ):

                feature_candidates.append(column)

        if len(feature_candidates) < 2:

            feature_candidates = numeric_columns.copy()

        target = st.selectbox(
            "🎯 Select Target Variable",
            feature_candidates
        )

        feature_columns = [
            col for col in feature_candidates
            if col != target
        ]

        if len(feature_columns) == 0:

            st.error(
                "No suitable feature variables available."
            )

        else:

            st.subheader("Features")

            selected_features = st.multiselect(
                "Select variables used for prediction",
                feature_columns,
                default=feature_columns
            )

            if len(selected_features) == 0:

                st.warning(
                    "Please select at least one feature."
                )

            else:

                ml_df = df[
                    selected_features + [target]
                ].dropna()

                if len(ml_df) < 10:

                    st.warning(
                        "At least 10 valid rows are recommended."
                    )

                else:

                    X = ml_df[selected_features]
                    y = ml_df[target]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.20,
                        random_state=42
                    )

                    # Linear Regression
                    linear_model = LinearRegression()

                    linear_model.fit(
                        X_train,
                        y_train
                    )

                    linear_prediction = linear_model.predict(
                        X_test
                    )

                    linear_r2 = r2_score(
                        y_test,
                        linear_prediction
                    )

                    linear_mae = mean_absolute_error(
                        y_test,
                        linear_prediction
                    )

                    # Random Forest
                    rf_model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=42
                    )

                    rf_model.fit(
                        X_train,
                        y_train
                    )

                    rf_prediction = rf_model.predict(
                        X_test
                    )

                    rf_r2 = r2_score(
                        y_test,
                        rf_prediction
                    )

                    rf_mae = mean_absolute_error(
                        y_test,
                        rf_prediction
                    )

                    st.subheader("Model Comparison")

                    comparison_df = pd.DataFrame({
                        "Model": [
                            "Linear Regression",
                            "Random Forest"
                        ],
                        "R² Score": [
                            linear_r2,
                            rf_r2
                        ],
                        "MAE": [
                            linear_mae,
                            rf_mae
                        ]
                    })

                    st.dataframe(
                        comparison_df,
                        use_container_width=True
                    )

                    # Select best model
                    if rf_r2 >= linear_r2:

                        best_model = rf_model
                        best_prediction = rf_prediction
                        best_name = "Random Forest"

                    else:

                        best_model = linear_model
                        best_prediction = linear_prediction
                        best_name = "Linear Regression"

                    st.success(
                        f"🏆 Best Model: {best_name}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "R² Score",
                            f"{max(linear_r2, rf_r2):.4f}"
                        )

                    with col2:

                        best_mae = (
                            rf_mae
                            if rf_r2 >= linear_r2
                            else linear_mae
                        )

                        st.metric(
                            "MAE",
                            f"{best_mae:.2f}"
                        )

                    st.subheader(
                        "Actual vs Predicted"
                    )

                    prediction_df = pd.DataFrame({
                        "Actual": y_test.values,
                        "Predicted": best_prediction
                    })

                    st.dataframe(
                        prediction_df.head(20),
                        use_container_width=True
                    )

                    # Feature importance
                    if isinstance(
                        best_model,
                        RandomForestRegressor
                    ):

                        st.subheader(
                            "Feature Importance"
                        )

                        importance_df = pd.DataFrame({
                            "Feature": selected_features,
                            "Importance": best_model.feature_importances_
                        }).sort_values(
                            "Importance",
                            ascending=False
                        )

                        st.dataframe(
                            importance_df,
                            use_container_width=True
                        )

                    # New prediction
                    st.subheader(
                        "🔮 Make a New Prediction"
                    )

                    input_values = {}

                    for feature in selected_features:

                        default_value = float(
                            df[feature].median()
                        )

                        input_values[feature] = st.number_input(
                            f"Enter {feature}",
                            value=default_value
                        )

                    if st.button(
                        "Predict"
                    ):

                        new_data = pd.DataFrame(
                            [input_values]
                        )

                        prediction = best_model.predict(
                            new_data
                        )[0]

                        st.success(
                            f"Predicted {target}: "
                            f"{prediction:.2f}"
                        )


# =========================================================
# 11. AUTOMATIC INSIGHTS
# =========================================================

elif section == "11. Automatic Insights":

    st.header("💡 Automatic Insights")

    st.subheader("Dataset")

    st.write(
        f"The dataset contains **{len(df)} rows** "
        f"and **{len(df.columns)} columns**."
    )

    if numeric_columns:

        st.subheader("Numeric Variables")

        for column in numeric_columns:

            mean_value = df[column].mean()
            max_value = df[column].max()
            min_value = df[column].min()

            st.write(
                f"**{column}** → "
                f"Mean: {mean_value:.2f}, "
                f"Min: {min_value:.2f}, "
                f"Max: {max_value:.2f}"
            )

    missing_total = int(
        df.isnull().sum().sum()
    )

    duplicate_total = int(
        df.duplicated().sum()
    )

    st.subheader("Data Quality Insights")

    if missing_total == 0:

        st.success(
            "✅ No missing values found."
        )

    else:

        st.warning(
            f"⚠️ {missing_total} missing values found."
        )

    if duplicate_total == 0:

        st.success(
            "✅ No duplicate rows found."
        )

    else:

        st.warning(
            f"⚠️ {duplicate_total} duplicate row(s) found."
        )

    if len(numeric_columns) >= 2:

        correlation = df[numeric_columns].corr()

        pairs = []

        for i in range(len(numeric_columns)):

            for j in range(i + 1, len(numeric_columns)):

                c1 = numeric_columns[i]
                c2 = numeric_columns[j]

                value = correlation.loc[c1, c2]

                if not pd.isna(value):

                    pairs.append(
                        (
                            c1,
                            c2,
                            value
                        )
                    )

        if pairs:

            strongest = max(
                pairs,
                key=lambda x: abs(x[2])
            )

            st.subheader(
                "Strongest Relationship"
            )

            st.write(
                f"**{strongest[0]} ↔ {strongest[1]}** "
                f"has correlation "
                f"**{strongest[2]:.3f}**."
            )


# =========================================================
# 12. DOWNLOAD CLEANED DATA
# =========================================================

elif section == "12. Download Cleaned Data":

    st.header("⬇️ Download Cleaned Dataset")

    cleaned_df = df.copy()

    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()

    # Numeric missing values
    for column in cleaned_df.select_dtypes(
        include=np.number
    ).columns:

        if cleaned_df[column].isnull().sum() > 0:

            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].median()
            )

    # Categorical missing values
    for column in cleaned_df.select_dtypes(
        exclude=np.number
    ).columns:

        if cleaned_df[column].isnull().sum() > 0:

            mode_value = cleaned_df[column].mode()

            if len(mode_value) > 0:

                cleaned_df[column] = cleaned_df[column].fillna(
                    mode_value.iloc[0]
                )

    st.success(
        "✅ Cleaned dataset is ready."
    )

    st.dataframe(
        cleaned_df.head(20),
        use_container_width=True
    )

    csv_data = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Mystery Dataset Analyzer | Algothon 26 Project"
)
