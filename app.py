from flask import Flask, request, render_template
import pandas as pd
import plotly.express as px
import plotly.io as pio

from classifier import classificar_medicamento

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        idade = int(request.form.get("idade"))
        sexo = int(request.form.get("sexo"))
        pressao = int(request.form.get("pressao"))
        potassio = float(request.form.get("potassio"))
        colesterol = int(request.form.get("colesterol"))

        resultado = classificar_medicamento(
            idade,
            sexo,
            pressao,
            colesterol,
            potassio
        )

        respostas = [2, 5, 1, 4, 1]

        dados = pd.DataFrame({
            'Remedio': ['Y', 'X', 'A', 'B', 'C'],
            'Valor': respostas
        })

        figura = px.bar(dados, x='Remedio', y='Valor')
        grafico = pio.to_html(figura, full_html=False)

        return render_template(
            'index.html',
            predicao=resultado,
            grafico=grafico
        )

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)