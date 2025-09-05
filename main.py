import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("Cleaned_DS_Jobs.csv")
    except:
        return None

# ----------------------------
# Preprocess Data
# ---------------------------
@st.cache_data
def preprocess_data(data: pd.DataFrame):
    """
    Creates:
    - df: analysis-ready subset of columns
    - model_data: encoded dataset for modeling
    """
    if data is None:
        return None, None

    # Keep only columns needed for analysis views
    keep_cols = [
        'Job Title', 'Rating', 'Company Name', 'Location', 'Size',
        'Type of ownership', 'Sector', 'Revenue', 'job_state',
        'avg_salary', 'company_age', 'python', 'excel', 'hadoop',
        'spark', 'aws', 'tableau', 'big_data', 'seniority'
    ]
    # errors='ignore' so it won't crash if a column is missing
    df = pd.DataFrame(data, columns=keep_cols).copy()

    # Clean/Map a few columns for consistent analysis
    if 'seniority' in df.columns:
        df['seniority'] = df['seniority'].map({'jr': 0, 'na': -1, 'senior': 1})
    if 'Sector' in df.columns:
        df['Sector'] = df['Sector'].astype(str).str.strip()
    if 'job_state' in df.columns:
        df['job_state'] = df['job_state'].astype(str).str.strip()

    # Build a modeling dataset: drop non-feature columns
    drop_cols = [
        'Salary Estimate', 'Job Description', 'Company Name', 'Location',
        'Headquarters', 'Size', 'Type of ownership', 'Industry', 'Revenue',
        'min_salary', 'max_salary', 'job_state', 'same_state',
        'company_age', 'job_simp'
    ]
    model_data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore').copy()

    # Encode categorical variables: seniority, Job Title, Sector
    if 'seniority' in model_data.columns:
        model_data['seniority'] = model_data['seniority'].map({'senior': 2, 'na': 0, 'jr': 1})

    # Use LabelEncoder safely (only if column exists)
    le_job = LabelEncoder()
    le_sector = LabelEncoder()

    if 'Job Title' in model_data.columns:
        model_data['Job Title'] = le_job.fit_transform(model_data['Job Title'].astype(str))
    if 'Sector' in model_data.columns:
        model_data['Sector'] = le_sector.fit_transform(model_data['Sector'].astype(str))

    return df, model_data

# ---------------------------
# Train Simple Models (as in your code)
# ---------------------------
@st.cache_data
def train_models(model_data: pd.DataFrame):
    """
    Trains Lasso, Ridge (regression) and DecisionTree/RandomForest (classification as in original).
    Returns models + splits for later inspection.
    """
    if model_data is None or 'avg_salary' not in model_data.columns:
        return None, None, None, None, None

    X = model_data.drop(columns=['avg_salary'], errors='ignore')
    y = model_data['avg_salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=30
    )

    models = {}

    # Lasso Regression
    lasso = Lasso(alpha=1.5)
    lasso.fit(X_train, y_train)
    models['Lasso'] = {
        'model': lasso,
        'train_score': lasso.score(X_train, y_train),
        'test_score': lasso.score(X_test, y_test)
    }

    # Ridge Regression
    ridge = Ridge(alpha=0.7)
    ridge.fit(X_train, y_train)
    models['Ridge'] = {
        'model': ridge,
        'train_score': ridge.score(X_train, y_train),
        'test_score': ridge.score(X_test, y_test)
    }

    # Decision Tree Classifier (kept same as your original)
    dt = DecisionTreeClassifier(max_depth=8)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = {
        'model': dt,
        'train_score': dt.score(X_train, y_train),
        'test_score': dt.score(X_test, y_test)
    }

    # Random Forest Classifier (kept same as your original)
    rf = RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = {
        'model': rf,
        'train_score': rf.score(X_train, y_train),
        'test_score': rf.score(X_test, y_test)
    }

    return models, X_train, X_test, y_train, y_test

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.title("📊 SalaryLens – Explore, Compare, Predict")

    uploaded_file = st.file_uploader("Upload Cleaned_DS_Jobs.csv", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        data = load_data()

    if data is None:
        st.warning("Please upload or place 'Cleaned_DS_Jobs.csv' in this folder.")
        return

    df, model_data = preprocess_data(data)

    # Sidebar
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Overview", "Salary Analysis", "Skills Analysis", "Company Analysis", "Job Search", "Prediction Models"]
    )

    # ---------------- Overview ----------------
    if page == "Overview":
        st.subheader("Dataset Overview")
        st.write(df.head())
        st.metric("Total Jobs", len(df))
        st.metric("Unique Companies", df["Company Name"].nunique() if "Company Name" in df.columns else 0)
        st.metric("Unique Titles", df["Job Title"].nunique() if "Job Title" in df.columns else 0)
        st.metric("Average Salary", f"${df['avg_salary'].mean():.0f}K" if "avg_salary" in df.columns else "N/A")

    # ---------------- Salary Analysis ----------------
    elif page == "Salary Analysis":
        st.subheader("Salary Distribution")
        if "avg_salary" in df.columns:
            fig = px.histogram(df, x="avg_salary", nbins=30, title="Salary Distribution")
            st.plotly_chart(fig)

        if "seniority" in df.columns:
            st.subheader("Average Salary by Seniority")
            seniority_map = {-1: "Unknown", 0: "Junior", 1: "Senior"}
            df["seniority_label"] = df["seniority"].map(seniority_map)
            fig2 = px.bar(df.groupby("seniority_label")["avg_salary"].mean().reset_index(),
                          x="seniority_label", y="avg_salary", title="Salary vs Seniority")
            st.plotly_chart(fig2)

    # ---------------- Skills Analysis ----------------
    elif page == "Skills Analysis":
        st.subheader("Skills Demand")
        skills = ["python", "excel", "hadoop", "spark", "aws", "tableau", "big_data"]
        available = [s for s in skills if s in df.columns]
        skill_counts = df[available].sum().sort_values(ascending=False)
        fig = px.bar(x=skill_counts.index, y=skill_counts.values, title="Skills Demand")
        st.plotly_chart(fig)

        st.subheader("Average Salary by Skill")
        skill_salary = []
        for skill in available:
            skill_jobs = df[df[skill] == 1]
            if len(skill_jobs) > 0:
                skill_salary.append({"Skill": skill, "Average Salary": skill_jobs["avg_salary"].mean()})
        if skill_salary:
            fig2 = px.bar(pd.DataFrame(skill_salary), x="Skill", y="Average Salary", title="Salary by Skills")
            st.plotly_chart(fig2)

    # ---------------- Company Analysis ----------------
    elif page == "Company Analysis":
        st.subheader("Top Rated Companies")
        if "Rating" in df.columns:
            top = df.sort_values("Rating", ascending=False).head(10)
            st.write(top[["Company Name", "avg_salary", "Rating"]])

        st.subheader("Company Ratings Distribution")
        if "Rating" in df.columns:
            fig = px.histogram(df, x="Rating", nbins=20, title="Company Ratings")
            st.plotly_chart(fig)

        if "Sector" in df.columns:
            st.subheader("Jobs by Sector")
            sector_counts = df["Sector"].value_counts().head(10)
            fig2 = px.pie(values=sector_counts.values, names=sector_counts.index, title="Top Sectors")
            st.plotly_chart(fig2)

    # ---------------- Job Search ----------------
    elif page == "Job Search":
        st.subheader("Filter Jobs")
        sector = st.selectbox("Sector", ["All"] + sorted(df["Sector"].dropna().unique()))
        state = st.selectbox("State", ["All"] + sorted(df["job_state"].dropna().unique()))

        filtered = df.copy()
        if sector != "All":
            filtered = filtered[filtered["Sector"] == sector]
        if state != "All":
            filtered = filtered[filtered["job_state"] == state]

        st.write(f"Found {len(filtered)} jobs")
        st.dataframe(filtered[["Job Title", "Company Name", "avg_salary", "Sector", "job_state"]].head(20))

    # ---------------- Prediction Models ----------------
    elif page == "Prediction Models":
        st.subheader("Model Performance")
        with st.spinner("Training..."):
            models, _, _, _, _ = train_models(model_data)
        if models:
            results = pd.DataFrame([
                            {"Model": k, "Train Score": v["train_score"], "Test Score": v["test_score"]}
                                for k, v in models.items()
                                                ])
            st.dataframe(results)
            fig = px.bar(results, x="Model", y=["Train Score", "Test Score"], barmode="group")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
