# Income Prediction

## Author
**Akash Thota**  
Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang  

- [GitHub Repository](#)  
- [LinkedIn Profile](#)  
- [PowerPoint Presentation](#)  
- [YouTube Video](#)  

---

## Background
### What is it about?
The project focuses on predicting whether an individual's income exceeds $50,000 based on various demographic and employment-related features. 

### Why does it matter?
This problem has practical applications in fields like personal finance, labor market analysis, and social studies. Accurate income prediction can help policymakers and businesses make informed decisions.

### Research Questions:
- What are the most significant factors influencing income?
- How accurately can machine learning models predict income levels based on demographic and employment-related data?

---

## Data
### Data Sources:
- UCI Machine Learning Repository

### Data Size:
- Approximately 48,842 rows and 15 columns

### Data Shape:
- Rows: 48,842
- Columns: 15

### Time Period:
- Not time-bound

### Row Representation:
- Each row represents an individual.

### Data Dictionary:
| Column Name        | Data Type | Definition                          | Potential Values             |
|--------------------|-----------|--------------------------------------|------------------------------|
| Age                | Integer   | Age of the individual               | 17 - 90                     |
| Workclass          | Categorical | Type of employment                | Private, Self-employed, etc. |
| Education          | Categorical | Level of education                | Bachelors, Masters, etc.     |
| Occupation         | Categorical | Type of job                       | Tech Support, Sales, etc.    |
| Hours per Week     | Integer   | Hours worked per week               | 1 - 99                      |
| Race               | Categorical | Race category                     | White, Black, etc.           |
| Gender             | Categorical | Gender of the individual          | Male, Female                 |
| Income (Target)    | Categorical | Income classification             | <=50K, >50K                  |

### Target Variable:
- **Income**: Categorized as `<=50K` or `>50K` (binary classification).

### Selected Features:
- Age, Education, Occupation, Hours per Week, Gender.

---

## Exploratory Data Analysis (EDA)
Performed data exploration using Jupyter Notebook:

- **Summary Statistics:**
  - Age: Normal distribution.
  - Hours worked per week: Skewed distribution.

- **Visualizations:**
  - Relationships between Education, Hours worked, and Income.

- **Data Cleansing:**
  - Missing values handled by dropping rows in critical columns like Workclass and Occupation.
  - Duplicates removed.

- **Data Transformation:**
  - Normalized numerical features using `MinMaxScaler`.
  - Categorical features encoded using `LabelEncoder`.

- **Class Imbalance:**
  - Identified imbalance in target variable (`<=50K` > `>50K`). Resampling techniques like SMOTE considered.

---

## Model Training
### Models Used:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost Classifier

### Training Approach:
- Train-Test Split: 80/20
- Hyperparameter Tuning: Used `GridSearchCV` for optimization.

### Development Environment:
- Python Packages: scikit-learn, XGBoost
- Environment: Google Colab

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- Best Model: XGBoost with 87.50% accuracy.

---

## Application of the Trained Models
Developed a web app using Streamlit:

- **Features:**
  - User Inputs: Age, Education, Occupation, Hours per Week, Gender.
  - Prediction Output: Income category (`<=50K` or `>50K`).

- **Benefits:**
  - Interactive and user-friendly interface for real-time predictions.

---

## Conclusion
### Summary:
- The XGBoost Classifier was the most accurate model with 87.50% accuracy.
- Key Features: Age, Education, Occupation, Hours worked per week, and Gender were critical in predicting income.

### Limitations:
- Imbalanced dataset.
- Limited to demographic and employment-related features.

### Lessons Learned:
- Feature selection and data preprocessing significantly impact model performance.

### Future Work:
- Experiment with advanced models like Neural Networks.
- Explore additional datasets to enhance predictions.

---

## References
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)
- Articles and blogs on XGBoost and income prediction techniques.
