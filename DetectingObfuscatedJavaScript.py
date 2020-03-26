import os
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

js_path = "C:\\Users\\tahsin.asif\\OneDrive - CYFIRMA INDIA PRIVATE LIMITED\\AI\\ObfuscatedJavaScriptData\\JavascriptSamples"
obfuscated_js_path = "C:\\Users\\tahsin.asif\\OneDrive - CYFIRMA INDIA PRIVATE LIMITED\\AI\\ObfuscatedJavaScriptData\\JavascriptSamplesObfuscated"

corpus = []
labels = []


file_types_and_labels = [(js_path, 0), (obfuscated_js_path, 1)]

for files_path, label in file_types_and_labels:
    files = os.listdir(files_path)
    for file in files:
        file_path = files_path + "/" + file
        try:
            with open(file_path, "r") as myfile:
                data = myfile.read().replace("\n", "")
                data = str(data)
                corpus.append(data)
                labels.append(label)
        except:
            pass


X_train, X_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.33, random_state=42
)

print(X_train)
text_clf = Pipeline(
    [
        ("vect", HashingVectorizer(input="content", ngram_range=(1, 3))),
        ("tfidf", TfidfTransformer(use_idf=True,)),
        ("rf", RandomForestClassifier(class_weight="balanced")),
    ]
)
text_clf.fit(X_train, y_train)

y_test_pred = text_clf.predict(X_test)

print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

