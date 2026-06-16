import pickle
import pandas
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

feature_names = vectorizer.get_feature_names_out()
log_probs = model.feature_log_prob_

# Verschil tussen subjectief en objectief per woord
verschil = log_probs[1] - log_probs[0]

zin = "Deze mens"
zin_als_getallen = vectorizer.transform([zin])

# Welke woorden zitten in deze zin
woorden = vectorizer.inverse_transform(zin_als_getallen)[0]

print("Woorden en hun invloed:")
for woord in woorden:
    index = vectorizer.vocabulary_[woord]
    score = verschil[index]
    richting = "→ subjectief" if score > 0 else "→ objectief"
    print(f"  {woord}: {score:.3f} {richting}")