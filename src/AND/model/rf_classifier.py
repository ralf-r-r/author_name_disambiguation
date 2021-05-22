from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)