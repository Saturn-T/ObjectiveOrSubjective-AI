import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

y=0

while y<=5:
    zin = input("Typ een zin: ")
    zin_als_getallen = vectorizer.transform([zin])
    voorspelling = model.predict(zin_als_getallen)

    if voorspelling[0] == 0:
        print("Dit is een objectieve zin")
    else:
        print("Dit is een subjectieve zin")
    y += 1