import pickle
import numpy as np

# Carrega o modelo uma única vez
model = pickle.load(open('models/modelo_preditivo.pkl', 'rb'))

def classificar_medicamento(idade, sexo, pressao, colesterol, potassio):

    caracteristicas = np.array([
        [idade, sexo, pressao, colesterol, potassio]
    ])

    predicao = model.predict(caracteristicas)

    mapeamento = {
        0: 'Y',
        1: 'X',
        2: 'A',
        3: 'B',
        4: 'C'
    }

    return mapeamento.get(predicao[0])