import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import statsmodels.api as sm

# Load your dataset
# df = pd.read_csv('your_dataset.csv')

# Define placeholders for column names
target_column = 'target'  # Column name for the target variable
numerical_columns = ['num_col1', 'num_col2']  # List of numerical columns
categorical_columns = ['cat_col1', 'cat_col2']  # List of categorical columns
columns_to_drop = ['unnecessary_col1', 'unnecessary_col2']  # List of columns to drop

# Descriptive Analysis
def descriptive_analysis(df):
    print("Descriptive Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

# Exploratory Analysis
def exploratory_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[numerical_columns[0]], kde=True, bins=30)
    plt.title(f'{numerical_columns[0]} Distribution')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x=categorical_columns[0], hue=target_column, data=df)
    plt.title(f'Survival Count by {categorical_columns[0]}')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Inferential Analysis
def inferential_analysis(df):
    group1 = df[df[target_column] == df[target_column].unique()[0]][numerical_columns[0]].dropna()
    group2 = df[df[target_column] == df[target_column].unique()[1]][numerical_columns[0]].dropna()
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print(f"T-Test between {numerical_columns[0]} of two groups")
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

# Predictive Analysis
def predictive_analysis(df):
    df = df.dropna(subset=numerical_columns + [target_column])
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    X = df.drop(columns=[target_column] + columns_to_drop)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))

# Causal Analysis
def causal_analysis(df):
    df = df.dropna(subset=numerical_columns + categorical_columns + [target_column])
    df['intercept'] = 1.0
    for col in categorical_columns:
        df[col] = (df[col] == df[col].unique()[1]).astype(float)
    
    logit_model = sm.Logit(df[target_column], df[['intercept'] + numerical_columns + categorical_columns])
    result = logit_model.fit()
    
    print("Logistic Regression Results:")
    print(result.summary())

# Run all analyses
def run_all_analyses(df):
    descriptive_analysis(df)
    exploratory_analysis(df)
    inferential_analysis(df)
    predictive_analysis(df)
    causal_analysis(df)

