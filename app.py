### FULL PROCESS DASHBOARD CODE:
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Performance - Advanced Dashboard", page_icon="🚀", layout="wide")

st.title("🚀 Student Performance Factors - Complete ML Process Dashboard")
st.markdown("**Data Cleaning → EDA → Model Training → Prediction | Live Process**")

# Load Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("StudentPerformanceFactors.xlsx")
        return df
    except:
        return None

df = load_data()

if df is None:
    st.error("❌ StudentPerformanceFactors.xlsx file Documents la illa da!")
    st.stop()

# Sidebar Process
st.sidebar.header("⚙️ PROCESS CONTROL")
process_step = st.sidebar.selectbox("Select Process Stage",
    ["1. Data Overview", "2. Data Cleaning Process", "3. EDA Process", "4. ML Model Training", "5. Live Prediction"])

st.sidebar.divider()
st.sidebar.write("**Dataset Info:**")
st.sidebar.write(f"Rows: {len(df)}")
st.sidebar.write(f"Columns: {len(df.columns)}")
st.sidebar.write(f"Missing: {df.isnull().sum().sum()}")

# 1. DATA OVERVIEW
if process_step == "1. Data Overview":
    st.header("📂 Step 1: Data Overview Process")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10), width='stretch')
    with col2:
        st.subheader("Data Types & Info")
        st.write(df.dtypes)
        st.write(f"**Shape:** {df.shape}")

    st.subheader("Missing Values Check (Process)")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning(f"Missing values found: {missing[missing>0]}")
        fig = px.bar(x=missing.index, y=missing.values, title="Missing Values per Column")
        st.plotly_chart(fig, width='stretch')
    else:
        st.success("✅ No Missing Values - Data Clean!")

# 2. DATA CLEANING PROCESS
elif process_step == "2. Data Cleaning Process":
    st.header("🧹 Step 2: Data Cleaning Process - LIVE")

    st.write("**Process Running...**")
    progress = st.progress(0)

    # Step 1: Remove duplicates
    progress.progress(20)
    st.write("🔄 Checking duplicates...")
    dup = df.duplicated().sum()
    st.write(f"Duplicates found: {dup}")
    df_clean = df.drop_duplicates()

    # Step 2: Handle missing
    progress.progress(50)
    st.write("🔄 Handling missing values...")
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)

    progress.progress(80)
    st.write("🔄 Encoding categorical values...")
    # Encoding
    le_dict = {}
    df_encoded = df_clean.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le

    progress.progress(100)
    st.success("✅ Cleaning Process Completed!")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before Cleaning:**", df.shape)
        st.dataframe(df.head(3))
    with col2:
        st.write("**After Cleaning:**", df_encoded.shape)
        st.dataframe(df_encoded.head(3))

    st.session_state['cleaned_df'] = df_encoded
    st.session_state['original_df'] = df_clean

# 3. EDA PROCESS
elif process_step == "3. EDA Process":
    st.header("📊 Step 3: EDA Process - Live Analysis")

    if 'original_df' in st.session_state:
        df_use = st.session_state['original_df']
    else:
        df_use = df

    tab1, tab2, tab3 = st.tabs(["📈 Correlation Process", "📊 Distribution Process", "🔍 Factor Impact"])

    with tab1:
        st.write("Calculating correlation matrix...")
        numeric_df = df_use.select_dtypes(include=np.number)
        corr = numeric_df.corr()
        st.write("**Top factors affecting Exam_Score:**")
        if 'Exam_Score' in corr.columns:
            top_corr = corr['Exam_Score'].sort_values(ascending=False)
            st.dataframe(top_corr)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    with tab2:
        col = st.selectbox("Select column for distribution", df_use.columns)
        fig = px.histogram(df_use, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, width='stretch')

        fig2 = px.box(df_use, y=col, title=f"Outlier Check - {col}")
        st.plotly_chart(fig2, width='stretch')

    with tab3:
        factor = st.selectbox("Select factor to see impact on Exam_Score",
                              [c for c in df_use.columns if c!= 'Exam_Score'])
        if df_use[factor].dtype == 'object':
            avg_score = df_use.groupby(factor)['Exam_Score'].mean().reset_index()
            fig = px.bar(avg_score, x=factor, y="Exam_Score", title=f"{factor} vs Exam_Score")
            st.plotly_chart(fig, width='stretch')
        else:
            fig = px.scatter(df_use, x=factor, y="Exam_Score", trendline="ols", title=f"{factor} vs Exam_Score")
            st.plotly_chart(fig, width='stretch')

# 4. ML MODEL TRAINING
elif process_step == "4. ML Model Training":
    st.header("🤖 Step 4: ML Model Training Process")

    if 'cleaned_df' not in st.session_state:
        st.warning("First Cleaning Process pannu da!")
        st.stop()

    df_encoded = st.session_state['cleaned_df']

    st.write("**Training Process Starting...**")

    # Features
    X = df_encoded.drop('Exam_Score', axis=1) if 'Exam_Score' in df_encoded.columns else df_encoded.iloc[:, :-1]
    y = df_encoded['Exam_Score'] if 'Exam_Score' in df_encoded.columns else df_encoded.iloc[:, -1]

    st.write(f"Features: {list(X.columns)}")
    st.write(f"Target: Exam_Score")

    test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.write(f"Training: {len(X_train)} | Testing: {len(X_test)}")

    model_choice = st.selectbox("Select Model", ["Random Forest", "Linear Regression"])

    if st.button("🚀 START TRAINING"):
        progress = st.progress(0)
        status = st.empty()

        status.write("🔄 Splitting data...")
        progress.progress(30)

        status.write("🔄 Training model...")
        progress.progress(60)

        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        progress.progress(80)

        status.write("🔄 Evaluating...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        progress.progress(100)
        status.write("✅ Training Completed!")

        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mse:.2f}")
        col2.metric("R2 Score", f"{r2:.4f}")
        col3.metric("Accuracy", f"{r2*100:.2f}%")

        # Feature importance
        if model_choice == "Random Forest":
            st.subheader("🔥 Feature Importance (Which factor matters most?)")
            importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
            fig = px.bar(importance, x="Importance", y="Feature", orientation='h', title="Feature Importance")
            st.plotly_chart(fig, width='stretch')

        # Actual vs Predicted
        st.subheader("Actual vs Predicted")
        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted Exam Score")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(dash="dash"))
        st.plotly_chart(fig, width='stretch')

        st.session_state['model'] = model
        st.session_state['features'] = list(X.columns)

# 5. LIVE PREDICTION
elif process_step == "5. Live Prediction":
    st.header("🎯 Step 5: Live Student Score Prediction")

    if 'model' not in st.session_state:
        st.warning("First Model Training pannu da!")
        st.stop()

    st.write("**Enter student details to predict Exam Score:**")

    model = st.session_state['model']
    features = st.session_state['features']

    # Get original df for input ranges
    orig_df = st.session_state.get('original_df', df)

    user_input = {}
    cols = st.columns(3)
    for i, feat in enumerate(features):
        with cols[i % 3]:
            if feat in orig_df.columns and orig_df[feat].dtype!= 'object':
                min_val = int(orig_df[feat].min())
                max_val = int(orig_df[feat].max())
                avg_val = int(orig_df[feat].mean())
                user_input[feat] = st.slider(f"{feat}", min_val, max_val, avg_val)
            else:
                user_input[feat] = st.number_input(f"{feat}", value=1)

    if st.button("🔮 PREDICT SCORE"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

        st.success(f"### 🎓 Predicted Exam Score: {prediction:.2f} / 100")

        if prediction >= 80:
            st.balloons()
            st.write("🌟 Excellent! High performer!")
        elif prediction >= 60:
            st.write("✅ Good! Pass!")
        else:
            st.write("⚠️ Needs improvement - More study hours needed!")

        # Gauge chart
        fig = px.bar(x=["Predicted Score"], y=[prediction], range_y=[0, 100], title="Score Gauge", color=[prediction], color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, width='stretch')
