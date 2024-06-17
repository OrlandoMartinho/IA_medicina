import joblib
from controller import controller

model_tratamento_calibrated = joblib.load(controller.models()[1])
model_hipotese_calibrated = joblib.load(controller.models()[0])

novo_sintoma = ["febre"]
previsao_tratamento = model_tratamento_calibrated.predict(novo_sintoma)
prob_tratamento = model_tratamento_calibrated.predict_proba(novo_sintoma).max()

previsao_hipotese = model_hipotese_calibrated.predict(novo_sintoma)
prob_hipotese = model_hipotese_calibrated.predict_proba(novo_sintoma).max()

print(f'Sintoma: {novo_sintoma}')
print(f'Previsão de Tratamento: {previsao_tratamento[0]} com {prob_tratamento * 100:.2f}% de confiança')
print(f'Previsão de Hipótese: {previsao_hipotese[0]} com {prob_hipotese * 100:.2f}% de confiança')
