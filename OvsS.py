import pickle
import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data = pandas.read_csv("zinnen.csv", sep=";")

zinnen = data["zin"]
labels = data["label"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(zinnen)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

model = MultinomialNB()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Nauwkeurigheid: {score * 100:.1f}%")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Model opgeslagen!")