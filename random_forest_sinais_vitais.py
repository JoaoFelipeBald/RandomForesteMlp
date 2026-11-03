"""Preve a gravidade e a classe a partir dos sinais vitais usando Random Forest."""

import math
import random
from pathlib import Path

PASTA = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA / '02_treino_sinais_vitais_com_label.txt'
ARQUIVO_VALIDACAO = PASTA / 'rf_previsoes_validacao_15.csv'
ARQUIVO_TESTE_CEGO = PASTA / 'rf_previsoes_teste_cego_15.csv'
SEMENTE = 42
FATOR_OVERSAMPLING_CLASSE_4 = 5  # Repete exemplos da classe 4 no treino.
ATRIBUTOS_POR_CORTE = 2
MIN_POR_FOLHA = 2
NUM_ARVORES = 100
PROFUNDIDADE_MAXIMA = 8
CANDIDATOS = [(NUM_ARVORES, PROFUNDIDADE_MAXIMA), (NUM_ARVORES, 6), (NUM_ARVORES, 10)]


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
                    f"Linha {numero_linha}: esperadas 8 colunas, recebidas {len(partes)}."
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


def encontrar_corte(amostras, sorteador):
    """Busca o corte que reduz o erro quadratico das folhas."""
    n = len(amostras)
    soma_total = sum(amostra[2] for amostra in amostras)
    quadrados_total = sum(amostra[2] ** 2 for amostra in amostras)
    menor_erro = quadrados_total - soma_total ** 2 / n
    melhor_corte = None

    for atributo in sorteador.sample(range(3), ATRIBUTOS_POR_CORTE):
        ordenadas = sorted(amostras, key=lambda item: item[1][atributo])
        soma_esquerda = 0.0
        quadrados_esquerda = 0.0
        for indice in range(n - 1):
            gravidade = ordenadas[indice][2]
            soma_esquerda += gravidade
            quadrados_esquerda += gravidade ** 2
            qtd_esquerda = indice + 1
            qtd_direita = n - qtd_esquerda
            if min(qtd_esquerda, qtd_direita) < MIN_POR_FOLHA:
                continue

            valor_atual = ordenadas[indice][1][atributo]
            valor_seguinte = ordenadas[indice + 1][1][atributo]
            if valor_atual == valor_seguinte:
                continue

            soma_direita = soma_total - soma_esquerda
            quadrados_direita = quadrados_total - quadrados_esquerda
            erro_esquerda = quadrados_esquerda - soma_esquerda ** 2 / qtd_esquerda
            erro_direita = quadrados_direita - soma_direita ** 2 / qtd_direita
            erro_total = erro_esquerda + erro_direita
            if erro_total < menor_erro - 1e-9:
                menor_erro = erro_total
                melhor_corte = (atributo, (valor_atual + valor_seguinte) / 2)
    return melhor_corte


def construir_arvore(amostras, profundidade, limite, sorteador):
    media = sum(item[2] for item in amostras) / len(amostras)
    if profundidade >= limite or len(amostras) < 2 * MIN_POR_FOLHA:
        return media
    corte = encontrar_corte(amostras, sorteador)
    if corte is None:
        return media
    atributo, valor = corte
    esquerda = [item for item in amostras if item[1][atributo] <= valor]
    direita = [item for item in amostras if item[1][atributo] > valor]
    return (
        atributo, valor,
        construir_arvore(esquerda, profundidade + 1, limite, sorteador),
        construir_arvore(direita, profundidade + 1, limite, sorteador),
    )


def treinar_floresta(treino, qtd_arvores, profundidade):
    sorteador = random.Random(SEMENTE)
    floresta = []
    for _ in range(qtd_arvores):
        # Seleciona amostras de treino com reposicao.
        amostra = [treino[sorteador.randrange(len(treino))] for _ in range(len(treino))]
        floresta.append(construir_arvore(amostra, 0, profundidade, sorteador))
    return floresta


def prever_arvore(arvore, entradas):
    while isinstance(arvore, tuple):
        atributo, corte, esquerda, direita = arvore
        arvore = esquerda if entradas[atributo] <= corte else direita
    return arvore


def prever_gravidade(floresta, entradas):
    return sum(prever_arvore(arvore, entradas) for arvore in floresta) / len(floresta)


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


def avaliar(floresta, dados, limites):
    previsoes = prever_registros(floresta, ocultar_respostas(dados), limites)
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


    print('\nrandom forest manual')
    print(f'dados: {len(treino)} treino, {len(validacao)} validação, '
          f'{len(teste_cego)} teste cego')
    print(f'oversampling classe 4: {FATOR_OVERSAMPLING_CLASSE_4}x')
    print('limites:', ', '.join(f'{valor:.4f}' for valor in limites))

    melhor_floresta = None
    melhor_parametros = None
    menor_rmse = float('inf')

    print('\nteste de parâmetros:')
    for qtd_arvores, profundidade in CANDIDATOS:
        floresta = treinar_floresta(treino_reforcado, qtd_arvores, profundidade)
        metricas = avaliar(floresta, validacao, limites)

        print(f"  {qtd_arvores} árvores, profundidade {profundidade} "
              f"| rmse {metricas['rmse']:.3f} | f1 {metricas['f1_macro']:.2%}")

        if metricas['rmse'] < menor_rmse:
            menor_rmse = metricas['rmse']
            melhor_floresta = floresta
            melhor_parametros = (qtd_arvores, profundidade)

    print(f'\nmelhor configuração: {melhor_parametros[0]} árvores, '
          f'profundidade {melhor_parametros[1]}')

    apresentar_metricas('treino', avaliar(melhor_floresta, treino, limites))

    previsoes_validacao = prever_registros(
        melhor_floresta, ocultar_respostas(validacao), limites
    )
    salvar_previsoes(ARQUIVO_VALIDACAO, previsoes_validacao)
    apresentar_metricas('validação',
                       avaliar_previsoes(validacao, previsoes_validacao))
    print(f'arquivo salvo: {ARQUIVO_VALIDACAO.name}')

    previsoes_teste_cego = prever_registros(
        melhor_floresta, ocultar_respostas(teste_cego), limites
    )
    salvar_previsoes(ARQUIVO_TESTE_CEGO, previsoes_teste_cego)
    apresentar_metricas('teste cego',
                       avaliar_previsoes(teste_cego, previsoes_teste_cego))
    print(f'arquivo salvo: {ARQUIVO_TESTE_CEGO.name}')



if __name__ == '__main__':
    main()
