import joblib
import pandas as pd

model = joblib.load(
    "models/salary_model.pkl"
)

sample = pd.DataFrame(
    {
        "Age":[30],
        "Gender":["Male"],
        "Education Level":["Master's"],
        "Job Title":["Data Analyst"],
        "Years of Experience":[5]
    }
)

salary = model.predict(sample)

print("Predicted Salary")

print(salary[0])