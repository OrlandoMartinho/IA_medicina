import pandas as pd
import joblib
from  controller import controller
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV




nome_arquivo = controller.data_set()
df = pd.read_csv(nome_arquivo)

colunas_interesse = ['mes', 'sintomas', 'tratamento', 'bairro', 'queixa', 'hipotese']
df = df[colunas_interesse].dropna()

X_train, X_test, y_train_tratamento, y_test_tratamento, y_train_hipotese, y_test_hipotese = train_test_split(
    df['sintomas'], df['tratamento'], df['hipotese'], test_size=0.2, random_state=42
)

model_tratamento = make_pipeline(CountVectorizer(), MultinomialNB())
model_tratamento_calibrated = CalibratedClassifierCV(model_tratamento, method='sigmoid')
model_tratamento_calibrated.fit(X_train, y_train_tratamento)

accuracy_tratamento = model_tratamento_calibrated.score(X_test, y_test_tratamento)
print(f'Acurácia do modelo de Tratamento: {accuracy_tratamento:.2f}')

model_hipotese = make_pipeline(CountVectorizer(), MultinomialNB())
model_hipotese_calibrated = CalibratedClassifierCV(model_hipotese, method='sigmoid')
model_hipotese_calibrated.fit(X_train, y_train_hipotese)

accuracy_hipotese = model_hipotese_calibrated.score(X_test, y_test_hipotese)
print(f'Acurácia do modelo de Hipótese: {accuracy_hipotese:.2f}')

joblib.dump(model_tratamento_calibrated,'../'+ controller.models()[1])
joblib.dump(model_hipotese_calibrated, '../'+controller.models()[0])
