# SalaryLens  

SalaryLens is a data science project built on **Glassdoor job postings**, exploring salary patterns and predicting pay ranges with machine learning.  

This repo has **two sides**:  

1. **Analysis** → Exploratory Data Analysis (EDA) to understand what drives salaries.  
2. **Prediction** → Machine learning models that try (and struggle!) to predict salaries using selected features.  

---

## 📂 Project Structure  

SalaryLens/
├── Data/
│ └── Cleaned_DS_Jobs.csv
├── Notebook/
│ ├── GlassDoor.ipynb # EDA: visuals and insights
│ └── salary_pre.ipynb # Modeling: prediction experiments & failures
├── requirements.txt # Python dependencies
├── README.md # You’re looking at it
└── main.py # Deployed live at salarylens.streamlit.app


---

## 🔎 Salary Analysis (EDA Notebook)  

- Explored **salary distributions** (min, max, avg).  
- Looked at trends by **sector, seniority, skills, and job title**.  
- Visualized boxplots and scatter plots.  
- **Key takeaway:** factors like **location, industry, and company size** dominate salary differences — but were excluded from the prediction notebook.  

---

## 🤖 Salary Prediction (Model Notebook)  

### Features Used  
Only a limited set of features were kept for modeling:  
- **Job Title** (label encoded)  
- **Sector** (label encoded)  
- **Seniority** (jr=1, senior=2, na=0)  
- **Skill Flags**: Python, Excel, Hadoop, Spark, AWS, Tableau, Big Data  

**Target:** `avg_salary`  

### Models Tried  
- Lasso, Ridge  
- Decision Tree, Random Forest    

**Current Result:** ~8% R² (low accuracy).  

---

## 🧠 Why Accuracy is Low  

- **Skills alone aren’t enough.** Job market pay is driven by **location, industry, company size, revenue**.    
- **Target (avg_salary)** is noisy — merging min/max salaries into one number adds error.  

---

## 🚀 Future Improvements  

### Better Features  
- Add back **location, company size, revenue, industry**.  
- Use **NLP embeddings** (TF-IDF/BERT) on job titles & descriptions.  

### Smarter Targets  
- Predict **min and max salaries separately**.  
- Or classify into **salary buckets** (low / medium / high).  

### More Data  
- Scrape more job postings for stronger signal.  

---

## 🌐 Deployment  
- Deployed on **Streamlit Cloud** → no local setup needed.  
- Live demo: [SalaryLens App](https://salarylens.streamlit.app/)  

---

## ✨ Takeaway  

- The **analysis** side gives valuable insights into salary patterns.  
- The **prediction** side shows current limitations (8% accuracy) and points toward improvements.  

---

## 🤝 Contributing  

Contributions are always welcome!  

If you spot a bug, want to suggest improvements, or add new features (like better models, richer features, or cooler visualizations), feel free to fork the repo and open a pull request.  

---

## 📌 Note  

This project is created **purely for learning purposes**.  
The models are experimental and not production-ready.  
The goal is to explore data cleaning, analysis, machine learning workflows, and deployment — not to provide accurate salary predictions.  

---

**SalaryLens is less about getting perfect predictions right now — and more about learning what really drives compensation.**  
