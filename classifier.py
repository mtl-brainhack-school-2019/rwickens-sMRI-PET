#written by atrophiedbrain - Joshua Morse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Unpickle data (load Python objects from disk)
with open("data.pkl", "rb") as f:
    ad_ages = pickle.load(f)
    ad_ct_cs = pickle.load(f)
    ad_ct_long = pickle.load(f)
    hc_ages = pickle.load(f)
    hc_ct_cs = pickle.load(f)
    hc_ct_long = pickle.load(f)

# We are going to build a random forest classifier between HC/AD using sklearn
# For this, we need to combine the cross-sectional data into one variable X
# And have labels for each subject in a variable y
# We will do this by creating two pandas DataFrames and then appending them together

# Create AD DataFrame, combining the cross-sectional regional CT data, ages, and a group label
ad_df = pd.DataFrame(data=ad_ct_cs)
ad_df['age'] = ad_ages
ad_df['AD'] = 1

# Create HC DataFrame, combining the cross-sectional regional CT data, ages, and a group label
hc_df = pd.DataFrame(data=hc_ct_cs)
hc_df['age'] = hc_ages
hc_df['AD'] = 0

# Append one list to another
data = ad_df.append(hc_df)

# Set X, y as required by our classifier
X = data[data.columns.difference(['AD'])]   # Features
y = data['AD']                              # Labels

# Split dataset into training set and test set, stratified by AD status
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Create a random forest classifier
# The random state is set for reproducibility
# Try changing the number of estimators and evaluating how this affects the accuracy of classification
classifier = RandomForestClassifier(n_estimators=2, random_state=34323)

# Train the classifier
classifier.fit(X_train, y_train)

# Obtain predictions from trained model
y_pred = classifier.predict(X_test)

# Calculate accuracy of prediction
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# List important features
feature_imp = pd.Series(classifier.feature_importances_, index=X.columns).sort_values(ascending=False)

# Visualize feature importance
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
