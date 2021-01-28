import pandas
from sklearn.tree import DecisionTreeClassifier

# Reads the CSV
df = pandas.read_csv("Machine Learning Test.csv")
# print(df)

# Coverts Dorm Yes or no To Binary Values
dorm = {'Yes': 1, 'No': 0}
df['Dorms'] = df['Dorms'].map(dorm)

# Coverts Major Yes or no To Binary Values
major = {'CS': 1, 'No': 0}
df['Major'] = df['Major'].map(major)

# Coverts Attending Yes or no To Binary Values
atten = {'Yes': 1, 'No': 0}
df['Attend'] = df['Attend'].map(atten)

# Main Features to look for
features = ['Distance', 'SchoolSize', 'Dorms', 'Major']
# Set features to find
X = df[features]
# Get whether or not to attend
y = df['Attend']

# Create decision tree
dtree = DecisionTreeClassifier().fit(X, y)

# Needs to be below
# ------ 1000 distance
# ------ Any size
# ------ Needs Dorms
# ------ Needs CS Major

# Print prediction
print(dtree.predict([[151, 49999, 1, 0]]))

print("[1] means 'Yes' attend")
print("[0] means 'No'")
