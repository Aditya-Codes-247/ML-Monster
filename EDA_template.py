# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import torch
import nltk
from bs4 import BeautifulSoup
import networkx as nx
import gensim
import lightgbm as lgb
import xgboost as xgb
import plotly.express as px
import bokeh

# Create a simple dataset
data = {
    'A': np.random.normal(0, 1, 100),
    'B': np.random.randint(0, 10, 100),
    'C': np.random.choice(['X', 'Y', 'Z'], 100),
    'D': np.random.uniform(1, 100, 100),
    'E': np.random.normal(50, 10, 100)
}
df = pd.DataFrame(data)

# Basic EDA
print("Data Shape:", df.shape)
print("Column Names:", df.columns)
print("Data Types:\n", df.dtypes)
print("Missing Values:\n", df.isnull().sum())

# Summary statistics
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Distribution plots
for column in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Boxplots
for column in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.show()

# Pairplot for pairwise relationships
sns.pairplot(df)
plt.title('Pairplot of Variables')
plt.show()

# Statistical tests (example: t-test)
sample1 = df['A']
sample2 = df['B']
t_stat, p_value = stats.ttest_ind(sample1, sample2)
print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

# Linear regression example
X = df[['E']]
y = df['D']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Natural Language Processing (example: NLTK)
nltk.download('punkt')
text = "This is a simple text for NLTK tokenization."
tokens = nltk.word_tokenize(text)
print("NLTK Tokenization:", tokens)

# Web scraping (example: BeautifulSoup)
html_content = "<html><body><h1>Hello, BeautifulSoup!</h1></body></html>"
soup = BeautifulSoup(html_content, 'html.parser')
print("BeautifulSoup Parsed Content:", soup.h1.text)

# Graphs and networks (example: NetworkX)
G = nx.Graph()
G.add_edge('Node1', 'Node2')
nx.draw(G, with_labels=True)
plt.title('NetworkX Graph')
plt.show()

# Word embeddings (example: Gensim)
sentences = [['This', 'is', 'sentence', 'one'], ['This', 'is', 'sentence', 'two']]
model = gensim.models.Word2Vec(sentences, min_count=1)
print("Word Embedding:", model['sentence'])

# Machine learning models (example: LightGBM)
train_data = lgb.Dataset(X_train, label=y_train)
params = {'objective': 'regression', 'metric': 'rmse'}
model = lgb.train(params, train_data, num_boost_round=100)
predictions = model.predict(X_test)
print("LightGBM Predictions:", predictions)

# Interactive plots (example: Plotly)
fig = px.scatter(df, x='A', y='B', color='C', size='E')
fig.update_layout(title='Plotly Scatter Plot')
fig.show()

# Additional visualizations (example: Bokeh)
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

source = ColumnDataSource(df)
p = figure(title="Bokeh Example", x_axis_label='X', y_axis_label='Y')
p.circle(x='A', y='B', source=source, size=10, color='navy', alpha=0.5)
show(p)
