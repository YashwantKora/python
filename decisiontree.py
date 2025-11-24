# 1️⃣ Import
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 2️⃣ Load
iris = load_iris()
X = iris.data
y = iris.target

# 3️⃣ Prepare
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 4️⃣ Model
model = DecisionTreeClassifier()

# 5️⃣ Train & Test
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
