import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime

# Função para carregar os dados
def load_data():
    url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"
    try:
        # Carregar os dados diretamente da URL
        tables = pd.read_html(url, encoding="latin1")
        data = tables[2]  # Pega a última tabela da página
        
        # Renomear as colunas para facilitar o processamento
        data.columns = ["Data", "Preco"]
        
        data["Preco"] = data["Preco"].ffill()
        # Filtrar apenas as linhas com valores válidos e criar uma cópia explícita
        data["Data"] = pd.to_datetime(data["Data"], format="%d/%m/%Y", dayfirst=True, errors="coerce")
        data = data[pd.to_datetime(data["Data"], errors="coerce").notnull()].copy()
        
        data.sort_values(by='Data', ascending=False, inplace=True) 
        data.set_index("Data", inplace=True)
        data.sort_index(inplace=True)
        
        # Converter a coluna 'Preco' para valores numéricos
        data["Preco"] = pd.to_numeric(data["Preco"], errors="coerce")
        
        # Remover linhas com valores ausentes
        data = data.dropna()
        
        return data
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

# Função para treinar o modelo preditivo
def train_model(data):
    data["Year"] = data.index.year
    data["Month"] = data.index.month
    data["Day"] = data.index.day

    X = data[["Year", "Month", "Day"]]
    y = data["Preco"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    return model, metrics

# Função para prever preços
def predict_price(model, year, month, date=None):
    if date is None:  # Previsão mensal
        days = [1, 15, 30]
        X_pred = pd.DataFrame({"Year": [year] * len(days), "Month": [month] * len(days), "Day": days})
        return model.predict(X_pred).mean()
    else:  # Previsão diária
        if isinstance(date, pd.Timestamp):  # Verifica se date é um objeto datetime
            X_pred = pd.DataFrame({"Year": [date.year], "Month": [date.month], "Day": [date.day]})
        elif isinstance(date, str):  # Caso a data seja uma string, converte para datetime
            parsed_date = pd.to_datetime(date, format='%d/%m/%Y')
            X_pred = pd.DataFrame({"Year": [parsed_date.year], "Month": [parsed_date.month], "Day": [parsed_date.day]})
        elif isinstance(date, datetime.date):  # Se date for um objeto datetime.date
            X_pred = pd.DataFrame({"Year": [date.year], "Month": [date.month], "Day": [date.day]})
        else:
            X_pred = pd.DataFrame({"Year": [year], "Month": [month], "Day": [date]})
        
        return model.predict(X_pred)[0]
