import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
data = pd.read_csv("/kaggle/input/insurance/insurance.csv")

# Basic info
data.info()
data.head()
data.describe()

df = pd.DataFrame(data)

# Unique values in each column
for column in df.columns:
    print(f"Unique values in {column} column:\n{df[column].value_counts()}\n")
    print("-" * 50)

# Check for missing values
df.isnull().sum()

# Distribution plots for age and charges
count, bin_edges = np.histogram(data["age"])
count2, bin_edges2 = np.histogram(data["charges"])

fig, axes = plt.subplots(1, 2, figsize=(17, 10))
axes[0].hist(df["age"], color="red", edgecolor="black")
axes[0].set_title("Distribution of Age", fontsize=20)
axes[0].set_xticks(bin_edges)
axes[1].hist(data["charges"], color="red", edgecolor="black")
axes[1].set_title("Distribution of Charges", fontsize=20)
axes[1].set_xticks(bin_edges2)
plt.tight_layout()
plt.show()

# Distribution of children
plt.title("Distribution of Number of Children")
data["children"].value_counts().plot(kind="bar", color="purple", rot=0)
plt.xlabel("Children")
plt.ylabel("Count")
plt.show()

# Pie charts for categorical features
fig, axes = plt.subplots(1, 3, figsize=(12, 10))
axes[0].pie(data["sex"].value_counts(), autopct='%1.1f%%', startangle=90)
axes[0].set_title("Sex", fontsize=18)
axes[0].legend(labels=data["sex"].value_counts().index, loc="upper right")
axes[1].pie(data["smoker"].value_counts(), autopct='%1.1f%%', startangle=90)
axes[1].set_title("Smoker", fontsize=20)
axes[1].legend(labels=data["smoker"].value_counts().index, loc="upper right")
axes[2].pie(data["region"].value_counts(), autopct='%1.1f%%', startangle=90)
axes[2].set_title("Region", fontsize=20)
axes[2].legend(labels=data["region"].value_counts().index, loc="upper right")
plt.tight_layout()
plt.show()

# Categorize BMI
data_copy = data.copy()
for i in range(len(data_copy)):
    bmi = data_copy.loc[i, "bmi"]
    if bmi < 18.5:
        data_copy.loc[i, "bmi"] = "Underweight"
    elif 18.5 <= bmi < 25:
        data_copy.loc[i, "bmi"] = "NormalWeight"
    elif 25 <= bmi < 30:
        data_copy.loc[i, "bmi"] = "Overweight"
    else:
        data_copy.loc[i, "bmi"] = "Obesity"

# Pie chart for BMI categories
fig, ax = plt.subplots(figsize=(8, 8))
plt.pie(data_copy["bmi"].value_counts(), autopct='%1.1f%%', startangle=90)
plt.title("BMI Category Distribution")
plt.legend(labels=data_copy["bmi"].value_counts().index, loc="upper right")
plt.show()

# 3D bar plot of average charges by age group and smoker status
bins = [18, 29, 39, 49, 59, 64]
labels = ['18-29', '30-39', '40-49', '50-59', '60-64']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
grouped = df.groupby(['age_group', 'smoker'])['charges'].mean().reset_index()
age_mapping = {label: i for i, label in enumerate(labels)}
smoker_mapping = {'no': 0, 'yes': 1}
x = [age_mapping[ag] for ag in grouped['age_group']]
y = [smoker_mapping[sm] for sm in grouped['smoker']]
z = np.zeros(len(grouped))
dx = dy = 0.5
dz = grouped['charges']
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(x, y, z, dx, dy, dz, color='tomato', shade=True)
ax.set_xlabel('Age Group')
ax.set_ylabel('Smoker')
ax.set_zlabel('Avg Cost')
ax.set_xticks(list(age_mapping.values()))
ax.set_xticklabels(list(age_mapping.keys()))
ax.set_yticks([0, 1])
ax.set_yticklabels(['No', 'Yes'])
plt.title('Average Insurance Cost by Age Group and Smoker Status')
plt.tight_layout()
plt.show()

# Outlier removal
z_scores = np.abs(zscore(data_copy[["age", "children"]]))
outliers = (z_scores > 3)
outliers_rows = outliers.any(axis=1)
df_outliers = data_copy[~outliers_rows]
outliers_count = data_copy.shape[0] - df_outliers.shape[0]
print(f"Number of outliers removed: {outliers_count}\n")

# One-Hot Encoding
categorical = df_outliers[["region", "smoker", "sex", "bmi"]]
ohe = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
ohe_array = ohe.fit_transform(categorical)
encoded_columns = ohe.get_feature_names_out(["region", "smoker", "sex", "bmi"])
encoded_df = pd.DataFrame(ohe_array, columns=encoded_columns)
encoded_df.index = df_outliers.index
df_encoded = pd.concat([df_outliers.drop(["region", "smoker", "sex", "bmi"], axis=1), encoded_df], axis=1)

# Feature interaction
df_encoded["age_charge_interaction"] = df_encoded["age"] * df_encoded["charges"]

# Correlation matrix
correlation_matrix = df_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Feature scaling
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df_encoded[["age", "children", "charges"]])
df_scaled = pd.DataFrame(scaled_array, columns=["age", "children", "charges"], index=df_encoded.index)
df_standardized = pd.concat([df_scaled, df_encoded.drop(["age", "children", "charges"], axis=1)], axis=1)

# Compare distribution before/after scaling
original_age = df_encoded['age']
standardized_age = df_standardized['age']
plt.figure(figsize=(10, 6))
sns.histplot(original_age, kde=True, color='green', label='Before Scaling', alpha=0.6)
sns.histplot(standardized_age, kde=True, color='purple', label='After Scaling', alpha=0.6)
plt.title('Age Distribution Before and After Standardization')
plt.legend()
plt.show()

# Model training
X = df_standardized.drop(columns=["charges"])
y = df_standardized["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Model Accuracy Metrics:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Coefficients
df_coef = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
df_coef_sorted = df_coef.sort_values("Coefficient")
plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficient", y=df_coef_sorted.index, data=df_coef_sorted, palette="coolwarm")
plt.title("Feature Importance (Coefficients)")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Prediction vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.title("Actual vs Predicted Charges (Test Set)")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.legend()
plt.show()

# Training vs test performance
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"{'Model Accuracy Comparison':^50}")
print(f"{'-'*50}")
print(f"{'Metric':<20}{'Train Set':<25}{'Test Set'}")
print(f"{'-'*50}")
print(f"{'MSE':<20}{train_mse:<25.2f}{test_mse:.2f}")
print(f"{'RMSE':<20}{train_rmse:<25.2f}{test_rmse:.2f}")
print(f"{'R² Score':<20}{train_r2:<25.4f}{test_r2:.4f}")
print(f"{'-'*50}\n")

def check_overfitting(train_rmse, test_rmse, train_r2, test_r2):
    if train_rmse < test_rmse and train_r2 > test_r2:
        return "Overfitting"
    else:
        return "No Overfitting"

overfitting_status = check_overfitting(train_rmse, test_rmse, train_r2, test_r2)
print(f"Overfitting Status: {overfitting_status}")

