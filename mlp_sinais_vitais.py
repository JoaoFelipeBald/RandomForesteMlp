"""Preve a gravidade e a classe a partir dos sinais vitais usando MLP."""

import math
import random
from pathlib import Path

PASTA = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA / '02_treino_sinais_vitais_com_label.txt'
ARQUIVO_VALIDACAO = PASTA / 'mlp_previsoes_validacao_15.csv'
ARQUIVO_TESTE_CEGO = PASTA / 'mlp_previsoes_teste_cego_15.csv'
SEMENTE = 42
FATOR_OVERSAMPLING_CLASSE_4 = 3  # Repete exemplos da classe 4 no treino.
TAXA_APRENDIZADO = 0.01
EPOCAS = 300
NEURONIOS_OCULTOS = 8
CANDIDATOS = [(NEURONIOS_OCULTOS, EPOCAS, TAXA_APRENDIZADO),
              (12, EPOCAS, TAXA_APRENDIZADO),
              (NEURONIOS_OCULTOS, 450, TAXA_APRENDIZADO)]


def ler_dados(caminho):
    """Le o dataset original com oito colunas."""
    dados = []
    identificadores = set()
    with open(caminho, encoding="utf-8-sig") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            if not linha.strip():
                continue
            partes = [item.strip() for item in linha.split(",")]
            if partes[0].lower() == "i":
                continue
            if len(partes) != 8:
                raise ValueError(
                    f"Linha {numero_linha}: precisa de 8 colunas, aqui tem {len(partes)}."
                )
            identificador = int(partes[0])
            identificadores.add(identificador)

            # Usa qPA, pulso e respiracao como entradas.
            entradas = [float(partes[3]), float(partes[4]), float(partes[5])]
            gravidade = float(partes[6])
            classe = int(partes[7])
            dados.append((identificador, entradas, gravidade, classe))

    return dados


def alocar_quantidades_por_classe(grupos, proporcao, quantidade_total):
    """Calcula quantos registros de cada classe entram no conjunto."""
    cotas = {classe: len(grupos[classe]) * proporcao for classe in grupos}
    quantidades = {classe: int(math.floor(cota)) for classe, cota in cotas.items()}
    restantes = quantidade_total - sum(quantidades.values())
    ordem = sorted(grupos, key=lambda classe: (-(cotas[classe] - quantidades[classe]), classe))
    for classe in ordem[:restantes]:
        quantidades[classe] += 1
    return quantidades


def separar_dados(dados):
    """Divide os registros em treino (70%), validacao (15%) e teste cego (15%)."""
    sorteador = random.Random(SEMENTE)
    grupos = {classe: [registro for registro in dados if registro[3] == classe]
              for classe in (1, 2, 3, 4)}
    for classe, grupo in grupos.items():
        sorteador.shuffle(grupo)

    # Separa 15% para comparar os hiperparametros.
    n_validacao = round(len(dados) * 0.15)
    validacao_por_classe = alocar_quantidades_por_classe(grupos, 0.15, n_validacao)
    validacao = []
    grupos_restantes = {}
    for classe, grupo in grupos.items():
        quantidade = validacao_por_classe[classe]
        validacao.extend(grupo[:quantidade])
        grupos_restantes[classe] = grupo[quantidade:]

    # Reserva outros 15% para o teste cego final.
    n_teste_cego = round(len(dados) * 0.15)
    proporcao = n_teste_cego / sum(len(grupo) for grupo in grupos_restantes.values())
    cego_por_classe = alocar_quantidades_por_classe(
        grupos_restantes, proporcao, n_teste_cego
    )
    treino, teste_cego = [], []
    for classe, grupo in grupos_restantes.items():
        quantidade = cego_por_classe[classe]
        teste_cego.extend(grupo[:quantidade])
        treino.extend(grupo[quantidade:])

    sorteador.shuffle(treino)
    sorteador.shuffle(validacao)
    sorteador.shuffle(teste_cego)
    return treino, validacao, teste_cego


def normalizar_entradas(entradas):
    """Coloca as entradas na escala usada pela rede."""
    qpa, pulso, respiracao = entradas
    return [qpa / 10.0, pulso / 100.0 - 1.0, respiracao / 11.0 - 1.0]


def criar_rede(qtd_neuronios):
    """Cria a rede com tres entradas, camada oculta e uma saida."""
    sorteador = random.Random(SEMENTE)
    return {
        "pesos_ocultos": [
            [sorteador.uniform(-0.5, 0.5) for _ in range(3)]
            for _ in range(qtd_neuronios)
        ],
        "bias_ocultos": [0.0] * qtd_neuronios,
        "pesos_saida": [sorteador.uniform(-0.5, 0.5) for _ in range(qtd_neuronios)],
        "bias_saida": 0.0,
    }


def feedforward(rede, entradas):
    x = normalizar_entradas(entradas)
    ocultos = []
    for pesos, bias in zip(rede["pesos_ocultos"], rede["bias_ocultos"]):
        soma = sum(pesos[i] * x[i] for i in range(3)) + bias
        ocultos.append(math.tanh(soma))
    saida = sum(peso * valor for peso, valor in zip(rede["pesos_saida"], ocultos))
    saida += rede["bias_saida"]
    return x, ocultos, saida  # Saida normalizada.


def backpropagation(rede, entradas, gravidade_real, taxa):
    x, ocultos, saida = feedforward(rede, entradas)
    erro = saida - gravidade_real / 100.0

    # Calcula os gradientes antes de atualizar os pesos.
    gradientes_ocultos = [
        erro * rede["pesos_saida"][j] * (1 - ocultos[j] ** 2)
        for j in range(len(ocultos))
    ]

    for j in range(len(ocultos)):
        rede["pesos_saida"][j] -= taxa * erro * ocultos[j]
    rede["bias_saida"] -= taxa * erro

    for j, gradiente in enumerate(gradientes_ocultos):
        for i in range(3):
            rede["pesos_ocultos"][j][i] -= taxa * gradiente * x[i]
        rede["bias_ocultos"][j] -= taxa * gradiente


def prever_gravidade(rede, entradas):
    _, _, saida = feedforward(rede, entradas)
    return saida * 100.0



def reforcar_classe_4(treino, fator=FATOR_OVERSAMPLING_CLASSE_4):
    """Repete registros da classe 4 no conjunto de treino."""
    classe_4 = [linha for linha in treino if linha[3] == 4]
    treino_reforcado = list(treino) + classe_4 * (fator - 1)
    random.Random(SEMENTE).shuffle(treino_reforcado)
    return treino_reforcado


def calcular_limites(treino):
    """Calcula os limites entre as classes a partir do treino."""
    limites = []
    for classe in (1, 2, 3):
        maior = max(linha[2] for linha in treino if linha[3] == classe)
        menor_seguinte = min(linha[2] for linha in treino if linha[3] == classe + 1)
        limites.append((maior + menor_seguinte) / 2)
    return limites


def classificar(gravidade, limites):
    for classe, limite in enumerate(limites, start=1):
        if gravidade < limite:
            return classe
    return 4


def calcular_rmse(rede, dados):
    return math.sqrt(sum(
        (prever_gravidade(rede, entradas) - gravidade) ** 2
        for _, entradas, gravidade, _ in dados
    ) / len(dados))


def treinar_rede(rede, treino, epocas, taxa):
    """Treina a rede com backpropagation e descida do gradiente."""
    sorteador = random.Random(SEMENTE)
    indices = list(range(len(treino)))
    historico = []
    for epoca in range(1, epocas + 1):
        sorteador.shuffle(indices)
        for indice in indices:
            _, entradas, gravidade, _ = treino[indice]
            backpropagation(rede, entradas, gravidade, taxa)
        if epoca == 1 or epoca % 50 == 0 or epoca == epocas:
            historico.append((epoca, calcular_rmse(rede, treino)))
    return historico


def avaliar_previsoes(registros_rotulados, previsoes):
    """Calcula as metricas comparando previsoes e valores reais."""
    resultados = {identificador: (gravidade, classe) for identificador, gravidade, classe in previsoes}
    matriz = [[0] * 4 for _ in range(4)]
    soma_erros_quadrados = 0.0
    for identificador, _, gravidade_real, classe_real in registros_rotulados:
        gravidade_prevista, classe_prevista = resultados[identificador]
        soma_erros_quadrados += (gravidade_prevista - gravidade_real) ** 2
        matriz[classe_real - 1][classe_prevista - 1] += 1

    rmse = math.sqrt(soma_erros_quadrados / len(registros_rotulados))
    acuracia = sum(matriz[k][k] for k in range(4)) / len(registros_rotulados)
    por_classe = []
    for k in range(4):
        vp = matriz[k][k]
        previstos = sum(matriz[j][k] for j in range(4))
        reais = sum(matriz[k])
        precision = vp / previstos if previstos else 0.0
        recall = vp / reais if reais else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        por_classe.append((precision, recall, f1, reais))

    return {
        "rmse": rmse,
        "acuracia": acuracia,
        "precision_macro": sum(item[0] for item in por_classe) / 4,
        "recall_macro": sum(item[1] for item in por_classe) / 4,
        "f1_macro": sum(item[2] for item in por_classe) / 4,
        "matriz": matriz,
        "por_classe": por_classe,
    }


def avaliar(rede, dados, limites):
    previsoes = prever_registros(rede, ocultar_respostas(dados), limites)
    return avaliar_previsoes(dados, previsoes)


def apresentar_metricas(nome, metricas):
    print(f'\n{nome}:')
    print(f"  rmse: {metricas['rmse']:.3f}")
    print(f"  acurácia: {metricas['acuracia']:.2%}")
    print(f"  precisão: {metricas['precision_macro']:.2%}")
    print(f"  recall: {metricas['recall_macro']:.2%}")
    print(f"  f1: {metricas['f1_macro']:.2%}")


def ocultar_respostas(teste):
    """Deixa apenas o identificador e as entradas."""
    return [(identificador, entradas) for identificador, entradas, _, _ in teste]



def prever_registros(modelo, teste_cego, limites):
    """Preve gravidade e classe usando as entradas."""
    resultado = []
    for identificador, entradas in teste_cego:
        gravidade = prever_gravidade(modelo, entradas)
        resultado.append((identificador, gravidade, classificar(gravidade, limites)))
    return resultado


def salvar_previsoes(caminho, previsoes):
    """Salva id, gravidade e classe em CSV."""
    with open(caminho, 'w', encoding='utf-8', newline='') as arquivo:
        arquivo.write('i,gravid,classe\n')
        for identificador, gravidade, classe in previsoes:
            arquivo.write(f'{identificador},{gravidade:.6f},{classe}\n')


def main():
    dados = ler_dados(ARQUIVO_DADOS)
    treino, validacao, teste_cego = separar_dados(dados)
    limites = calcular_limites(treino)
    treino_reforcado = reforcar_classe_4(treino)

    print('\nmlp manual')
    print(f'dados: {len(treino)} treino, {len(validacao)} validação, '
          f'{len(teste_cego)} teste cego')
    print(f'oversampling classe 4: {FATOR_OVERSAMPLING_CLASSE_4}x')
    print('limites:', ', '.join(f'{valor:.4f}' for valor in limites))

    melhor_rede = None
    melhor_parametros = None
    melhor_historico = None
    menor_rmse = float('inf')

    print('\nteste de parâmetros:')
    for neuronios, epocas, taxa in CANDIDATOS:
        rede = criar_rede(neuronios)
        historico = treinar_rede(rede, treino_reforcado, epocas, taxa)
        metricas = avaliar(rede, validacao, limites)

        print(f"  {neuronios} neurônios, {epocas} épocas, taxa {taxa} "
              f"| rmse {metricas['rmse']:.3f} | f1 {metricas['f1_macro']:.2%}")

        if metricas['rmse'] < menor_rmse:
            menor_rmse = metricas['rmse']
            melhor_rede = rede
            melhor_parametros = (neuronios, epocas, taxa)
            melhor_historico = historico

    print(f'\nmelhor configuração: {melhor_parametros[0]} neurônios, '
          f'{melhor_parametros[1]} épocas, taxa {melhor_parametros[2]}')

    apresentar_metricas('treino', avaliar(melhor_rede, treino, limites))

    previsoes_validacao = prever_registros(
        melhor_rede, ocultar_respostas(validacao), limites
    )
    salvar_previsoes(ARQUIVO_VALIDACAO, previsoes_validacao)
    apresentar_metricas('validação',
                       avaliar_previsoes(validacao, previsoes_validacao))
    print(f'arquivo salvo: {ARQUIVO_VALIDACAO.name}')

    previsoes_teste_cego = prever_registros(
        melhor_rede, ocultar_respostas(teste_cego), limites
    )
    salvar_previsoes(ARQUIVO_TESTE_CEGO, previsoes_teste_cego)
    apresentar_metricas('teste cego',
                       avaliar_previsoes(teste_cego, previsoes_teste_cego))
    print(f'arquivo salvo: {ARQUIVO_TESTE_CEGO.name}')


if __name__ == '__main__':
    main()
