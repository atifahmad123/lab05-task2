import pandas as pd
import matplotlib.pyplot as plt


# Create 12 rows: two missing Marks values and one exact duplicate.
data = {
    "StudentID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1],
    "Gender": ["Female", "Male", "Female", "Male", "Female", "Male",
               "Female", "Male", "Female", "Male", "Female", "Female"],
    "StudyHours": [5, 3, 6, 2, 7, 4, 5, 1, 8, 3, 6, 5],
    "Attendance(%)": [90, 80, 92, 75, 95, 85, 88, 60, 98, 82, 91, 90],
    "Marks": [78, 65, 85, 55, 90, None, 75, 30, 95, 68, None, 78],
}
pd.DataFrame(data).to_csv("student_performance.csv", index=False)

# Load and inspect the dataset.
df = pd.read_csv("student_performance.csv")
print("Shape:", df.shape)
df.info()
print("\nMissing values per column:")
print(df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicates before calculating the median.
df = df.drop_duplicates().copy()
df["Marks"] = df["Marks"].fillna(df["Marks"].median())
print("\nAfter cleaning:", df.shape)

plt.figure()
plt.hist(df["Marks"], bins=6, edgecolor="black")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.title("Distribution of Student Marks")
plt.tight_layout()
plt.savefig("marks_histogram.png")
plt.close()

plt.figure()
plt.boxplot(df["Marks"])
plt.xticks([1], ["Marks"])
plt.ylabel("Marks")
plt.title("Boxplot of Student Marks")
plt.tight_layout()
plt.savefig("marks_boxplot.png")
plt.close()

# Check for outliers using the same IQR rule as the boxplot.
q1 = df["Marks"].quantile(0.25)
q3 = df["Marks"].quantile(0.75)
iqr = q3 - q1
outliers = df[(df["Marks"] < q1 - 1.5 * iqr)
              | (df["Marks"] > q3 + 1.5 * iqr)]
print("\nMarks outliers (IQR rule):")
print(outliers if not outliers.empty else "No outliers found.")

plt.figure()
plt.scatter(df["StudyHours"], df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs. Marks")
plt.tight_layout()
plt.savefig("studyhours_vs_marks.png")
plt.close()

# Full numeric correlation matrix, including the numeric StudentID column.
print("\nCorrelation matrix:")
print(df.corr(numeric_only=True))

print("\nAverage Marks and Attendance by Gender:")
print(df.groupby("Gender")[["Marks", "Attendance(%)"]].mean())

print("\nSaved marks_histogram.png, marks_boxplot.png, studyhours_vs_marks.png")
