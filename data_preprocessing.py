import pandas as pd

# Load dataset
try:
    df = pd.read_csv("medical_insurance.csv")
except FileNotFoundError:
    print("❌ CSV file not found. Please check path and filename.")
    exit()

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode categorical features
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Feature engineering: BMI category
df_encoded['bmi_category'] = pd.cut(df_encoded['bmi'],
                                    bins=[0, 18.5, 25, 30, 100],
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

df_encoded = pd.get_dummies(df_encoded, columns=['bmi_category'], drop_first=True)

# Save clean data
df_encoded.to_csv("cleaned_medical_insurance.csv", index=False)
print("✅ Data preprocessing complete.")