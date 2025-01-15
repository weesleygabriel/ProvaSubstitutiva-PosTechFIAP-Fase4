import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from pipeline import load_data, train_model, predict_price
from fpdf import FPDF
import os
import base64
import time
import tempfile

# Função para exportar o PDF
def export_pdf(content, filename, img_path):
    try:
        # Criação do PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Relatório de Previsão do Preço do Petróleo Brent", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Previsão realizada com base nos dados históricos do IPEADATA", ln=True, align='C')
        pdf.ln(10)  

        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)

        # Inserindo o gráfico no PDF como imagem
        if os.path.exists(img_path):
            pdf.image(img_path, x=10, y=pdf.get_y(), w=190)
        else:
            print(f"Imagem não encontrada: {img_path}")

        pdf_output_path = os.path.abspath(filename)
        pdf.output(pdf_output_path)
        return pdf_output_path, None 
    except Exception as e:
        return "Finalizado.", str(e)

# Função para iniciar o download caso não tenha feito automaticamente
def download_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="{os.path.basename(file_path)}">Se o download não iniciou automaticamente, clique aqui para realizar o download novamente.</a>'
        st.markdown(href, unsafe_allow_html=True)


data = load_data()

if not data.empty:
    st.title("Previsão do Preço do Petróleo 🛢️")
    st.write("Utilize esta aplicação para prever o preço do petróleo Brent com base nos dados históricos do IPEADATA.")
    st.subheader("Dados Históricos")
    st.write(data.tail(10))

    # Plot histórico do último ano
    st.subheader("Evolução do Preço - Último Ano")
    last_year_data = data[data.index.year == data.index.year.max() - 1]


    monthly_mean = last_year_data.resample('ME').mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(monthly_mean.index, monthly_mean["Preco"], width=0.4, label="Preço Médio Mensal")
    plt.title("Evolução do Preço no Último Ano")
    plt.xlabel("Mês")
    plt.ylabel("Preço (USD)")
    ax.legend()
    plt.xticks(rotation=90)
    ax.set_xticks(monthly_mean.index)
    ax.set_xticklabels([d.strftime('%m/%Y') for d in monthly_mean.index], rotation=90)
    ax.bar_label(bars, fmt="%.01f", size=10, label_type="edge")

    plt.tight_layout()
    st.pyplot(fig)

    # Treinamento do modelo preditivo
    st.subheader("Treinamento do Modelo")
    model, metrics = train_model(data)
    st.write("Desempenho do Modelo:")
    st.write(metrics)

    # Escolha de previsão
    st.subheader("Escolha o Tipo de Previsão")
    option = st.radio("Selecione:", ("Média Mensal", "Preço Diário"))


    if option == "Média Mensal":
        year = st.number_input("Ano", min_value=1987, max_value=2025, step=1)
        month = st.selectbox("Mês", list(calendar.month_name)[1:])

        if st.button("Prever"):
            month_num = list(calendar.month_name).index(month)
            avg_price = predict_price(model, year, month_num)
            
            st.session_state['avg_price'] = avg_price
            st.session_state['month'] = month
            st.session_state['year'] = year

            st.success(f"O preço médio previsto do petróleo Brent para o mês {month}/{year} é de ${avg_price:.2f} por barril.")

            
            local_plot_path = "temp_plot.png"
            future_months = pd.date_range(start=f"{year}-{month_num}-01", periods=(pd.Timestamp(f"{year}-{month_num}-01").days_in_month), freq="D")
            future_prices = [predict_price(model, d.year, d.month, d.day) for d in future_months]
            plt.figure(figsize=(10, 5))
            ax = plt.gca()
            ax.plot(future_months, future_prices, label="Previsão", marker='o', color='orange')
            plt.title(f"📊 Evolução do Preço em {month}/{year}")
            plt.xlabel("Data")
            plt.ylabel("Preço (USD)")
            plt.gcf().autofmt_xdate()
            ax.set_xticks(future_months)
            ax.set_xticklabels([d.strftime('%d/%m/%Y') for d in future_months], rotation=90)
            plt.tight_layout()
            plt.savefig(local_plot_path)
            st.pyplot(plt)
            

        if 'avg_price' in st.session_state and st.button("Exportar"):
            text = f"O preço médio previsto do petróleo Brent para o mês {st.session_state['month']}/{st.session_state['year']} é de ${st.session_state['avg_price']:.2f} por barril."
            path = f'previsao_mensal_{st.session_state["month"]}_{st.session_state["year"]}.pdf'
            imagem = "temp_plot.png"
            pdf_path, error = export_pdf(text, path, imagem)
            if pdf_path:
                status_message = st.success("Iniciando download...")
                time.sleep(5)
                status_message.empty() 
                st.write("")
                status_message = st.success("O download foi realizado com sucesso!")
                time.sleep(8)
                status_message.empty() 
                st.write("")
                download_pdf(pdf_path)
            else:
                st.error(f"Falha ao gerar o PDF. Motivo: {error}")

    elif option == "Preço Diário":
        date = st.date_input("Informe a data que deseja prever")

        if st.button("Prever"):
            daily_price = predict_price(model, date.year, date.month, date.day)
            st.session_state['daily_price'] = daily_price
            st.session_state['date'] = date

            st.success(f"O preço previsto do petróleo Brent para o dia {date.day}/{date.month}/{date.year} é de ${daily_price:.2f} por barril.")

            local_plot_path = "temp_plot.png"
            future_days = pd.date_range(start=f"{date.year}-{date.month}-01", periods=(pd.Timestamp(date).days_in_month), freq="D")
            future_prices = [predict_price(model, d.year, d.month, d.day) for d in future_days]
            plt.figure(figsize=(10, 5))
            ax = plt.gca()
            ax.plot(future_days, future_prices, label="Previsão", marker='o', color='purple')
            plt.title(f"📆 Evolução do Preço Diário em {date.month}/{date.year}")
            plt.xlabel("Data")
            plt.ylabel("Preço (USD)")
            plt.gcf().autofmt_xdate()
            ax.set_xticks(future_days)
            ax.set_xticklabels([d.strftime('%d/%m/%Y') for d in future_days], rotation=90)
            plt.tight_layout()
            plt.savefig(local_plot_path)
            st.pyplot(plt)
            

        if 'daily_price' in st.session_state and st.button("Exportar"):
            
            text = f"O preço previsto do petróleo Brent para o dia {st.session_state['date'].day}/{st.session_state['date'].month}/{st.session_state['date'].year} é de ${st.session_state['daily_price']:.2f} por barril."
            path = f'previsao_diaria_{st.session_state["date"].day}_{st.session_state["date"].month}_{st.session_state["date"].year}.pdf'
            imagem = "temp_plot.png"
            pdf_path, error = export_pdf(text, path, imagem)
            if pdf_path:
                status_message = st.success("Iniciando download...")
                time.sleep(5)
                status_message.empty() 
                st.write("")
                status_message = st.success("O download foi realizado com sucesso!")
                time.sleep(8)
                status_message.empty() 
                st.write("")
                download_pdf(pdf_path)
            else:
                st.error(f"Falha ao gerar o PDF. Motivo: {error}")

    st.markdown("---")
    st.markdown("Powered by Wesley Fonseca")
