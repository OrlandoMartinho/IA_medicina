import pandas as pd
import matplotlib.pyplot as plt

nome_arquivo = 'forhealth_data.csv'

df = pd.read_csv(nome_arquivo)
print(df.describe())
contagem_por_mes = df['hipotese'].value_counts()

plt.figure(figsize=(10, 6))
contagem_por_mes.plot(kind='bar', color='skyblue')
plt.title('Número de casos de Hanseníase por mês')
plt.xlabel('Mês')
plt.ylabel('Número de casos')
plt.show()

