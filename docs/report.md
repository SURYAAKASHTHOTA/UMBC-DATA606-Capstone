# Income Prediction - Combined Report

## Author
**Akash Thota**  
**Semester:** Fall'24  
**Prepared for:** UMBC Data Science Master's Degree Capstone by Dr. Chaojie (Jay) Wang  

- **GitHub Repository:** [UMBC-DATA606-Capstone](https://github.com/AkashThota/UMBC-DATA606-Capstone)  
- **LinkedIn Profile:** [Akash Thota LinkedIn](https://www.linkedin.com/in/akash-thota-719030296)  

---

## 1. Background

### What is the project about?
This project focuses on predicting an individual's income category (greater than or less than $50K) based on the Adult dataset, which contains various demographic and work-related attributes. The goal is to use machine learning models to understand which factors most significantly impact income levels and accurately classify individuals into the appropriate income category.

### Why is it important?
Understanding income prediction can have several practical applications, including tax policy analysis, wage gap identification, and targeted economic interventions. It can also assist in policy-making decisions that aim to address inequality and improve social welfare. Accurately predicting income levels can provide insights into how factors such as education, occupation, and age influence income.

### Research Questions:
1. **Which factors have the most impact on income prediction?**
   - This question examines the influence of attributes like education, occupation, and hours worked per week on income levels.

2. **Can machine learning accurately predict income levels?**
   - This explores the performance of various machine learning models in classifying income.

3. **Which machine learning model performs the best for income prediction?**
   - A comparison of different models (Logistic Regression, Random Forest, XGBoost) to determine the most effective one based on metrics like accuracy, precision, and F1-score.

4. **Does including demographic features such as race and gender improve prediction accuracy?**
   - This will investigate whether these demographic variables enhance model performance and whether their influence is statistically significant.

---

## 2. Data

### Dataset Overview
The dataset used in this project is the Adult dataset, which was extracted from the UCI Machine Learning Repository. It includes data on various personal attributes such as age, education, marital status, and occupation, along with their corresponding income class (`<=50K` or `>50K`).

### Data Sources
- **Source:** UCI Machine Learning Repository  
- **Files:** `adult.csv`

### Data Size
- `adult.csv`: [3.9 MB]

### Data Shape
- **Total instances:** 48,842 rows, 15 columns (after removing missing values, there are 45,222 rows)

### Time Period
The dataset is derived from the 1994 US Census.

### Row Representation
Each row represents a single individual with attributes such as their age, education, marital status, and occupation, along with their corresponding income class (`<=50K` or `>50K`).

### Data Dictionary
| Column Name      | Data Type    | Definition                          | Potential Values           |
|------------------|--------------|--------------------------------------|----------------------------|
| Age              | Numerical    | Age of the individual               | Numeric values (e.g., 25)  |
| Workclass        | Categorical  | Employment type                     | Private, Self-emp, etc.    |
| fnlwgt           | Numerical    | Census weighting factor             | Numeric values             |
| Education        | Categorical  | Education level                     | Bachelors, HS-grad, etc.   |
| Education-num    | Numerical    | Number of years of education        | Numeric values             |
| Marital-status   | Categorical  | Marital status                      | Married, Divorced, etc.    |
| Occupation       | Categorical  | Type of job                         | Exec-managerial, Sales     |
| Relationship     | Categorical  | Family relationship                 | Husband, Wife, etc.        |
| Race             | Categorical  | Race category                       | White, Black, etc.         |
| Sex              | Categorical  | Gender                              | Male, Female               |
| Capital-gain     | Numerical    | Income from investment sources      | Numeric values             |
| Capital-loss     | Numerical    | Loss from investment sources        | Numeric values             |
| Hours-per-week   | Numerical    | Hours worked per week               | Numeric values             |
| Native-country   | Categorical  | Country of origin                   | United States, etc.        |
| Income           | Categorical  | Income class (Target variable)      | `<=50K`, `>50K`            |

### Target Variable:
- **Income**: Categorized as `<=50K` or `>50K` (binary classification).

### Selected Features:
- Age, Education, Occupation, Hours per Week, Gender.

---

## 3. Data Preprocessing

### Preprocessing Steps:
1. **Encoding Categorical Features:**
   - Used one-hot encoding for features like Workclass, Occupation, etc.
   - Label encoding for Gender and Race.
2. **Handling Missing Data:**
   - Dropped rows with missing values in critical columns like Workclass and Occupation.
3. **Scaling Numerical Features:**
   - Applied `MinMaxScaler` for uniform scaling of numerical data like Age and Hours per Week.
4. **Addressing Class Imbalance:**
   - Applied SMOTE to balance classes in the target variable.

---

## 4. Exploratory Data Analysis (EDA)

### Visualizing Relationships Between Variables
Explored the relationships between various features and the target variable (`Income`). Features such as Education, Occupation, and Hours worked showed significant influence on income levels.

### Boxplot for Outlier Detection
Outliers were identified in Age and Hours worked per week using boxplots.

### Distribution of Numeric Features
- Age follows a normal distribution.
- Hours worked per week is skewed, requiring potential transformation.

### Distribution of Income
Identified imbalance with a majority of individuals falling into the `<=50K` category.

---

## 5. Model Training

### Models Used:
1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)
4. XGBoost Classifier

### Training Approach:
- Train-Test Split: 80/20
- Hyperparameter Tuning: `GridSearchCV` for optimization

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score

### Best Model:
- XGBoost Classifier with 87.50% accuracy.

---

## 6. Application of Trained Models

### Streamlit App:
Developed an interactive Streamlit app to allow users to input data and predict income class in real-time.

---

## 7. Conclusion

### Summary:
- The XGBoost Classifier demonstrated the highest performance.
- Critical features: Age, Education, Occupation, Hours per Week, Gender.

### Limitations:
- Imbalanced dataset.
- Limited features from the dataset.

### Future Work:
- Experiment with Neural Networks.
- Explore additional datasets for enhanced insights.

---

## References
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)
- Articles and blogs on XGBoost and income prediction techniques.
