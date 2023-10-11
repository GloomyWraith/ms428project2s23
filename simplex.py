# Pacotes necessários
import numpy as np

# Leitura dos dados

linhas = int(input("Informe o tamanho da matriz.\nLinhas: "))
colunas = int(input("Colunas: "))

A = np.zeros(linhas*colunas, dtype=float).reshape(linhas, colunas) # Define a matriz 

print("Entre a matriz dos coeficientes das restrições, linha por linha, com os coeficientes separados por espaço, já incluindo as variáveis de folga.")

# Realiza a leitura da matriz, linha por linha
for i in range(linhas): 
    A[i] = input().split() 


# Lê o vetor dos custos
print("Informe o vetor dos custos (elementos separados por um espaço): ")
custos = [float(c) for c in input().split(sep = " ")]

# Lê o vetor dos recursos (vetor b)
print("Informe o vetor dos recursos (elementos separados por um espaço): ")
recursos = [float(b) for b in input().split(sep = " ")]


# Define partições e os custos básicos e não-básicos iniciais
particao_basica = A[:,(linhas-colunas):].copy()
particao_nbasica = A[:,:linhas].copy()
custos_basicos = custos[(linhas-colunas):].copy()
custos_nbasicos = custos[:linhas].copy()

k = 0 # Define contador

while True:
    # Calcula a solução básica factível Xb
    xb = np.linalg.solve(particao_basica, recursos)

    # Calcula o valor atual da função
    valor_atual = np.dot(np.transpose(custos_basicos), xb)
    
    # Calcula o multiplicador Simplex
    lambda_simplex = np.linalg.solve(np.transpose(particao_basica), custos_basicos)
    
    # Calcula os custos relativos e armazena o indice de qual variável deve entrar na base
    custos_relativos = [(custos_nbasicos[j] - np.dot(np.transpose(lambda_simplex), particao_nbasica[:,j])) for j in range(colunas-linhas)]
    custo_relativo_min = min(custos_relativos)
    indice_entrada_base = custos_relativos.index(custo_relativo_min)

    # Teste de otimalidade
    if (custo_relativo_min >= 0):
        print("Solução atual é ótima.")
        print("Valor ótimo da solução: ", valor_atual)
        print("Variáveis ótimas: ", xb)
        break
    else:
        # Cálculo da Direção Simplex
        direcao_simplex = np.linalg.solve(particao_basica, particao_nbasica[:,[indice_entrada_base]])

        if True in (direcao_simplex <= 0):
            print("Problema não tem solução ótima finita.")
            break
        else:
            passo = xb/direcao_simplex
            epsilon = min(passo)
            indice_saida_base = np.where(passo == epsilon)[0][0]
    
    # Atualiza partições básica e não-básica
    entrada_base = particao_nbasica[:,indice_entrada_base].copy()
    saida_base = particao_basica[:,indice_saida_base].copy()
    particao_basica[:,indice_saida_base] = entrada_base
    particao_nbasica[:,indice_entrada_base] = saida_base

    # Atualiza custos básicos e não-básicos
    entrada_custo = custos_nbasicos[indice_entrada_base]
    saida_custo = custos_basicos[indice_saida_base]
    custos_basicos[indice_saida_base] = entrada_custo
    custos_nbasicos[indice_entrada_base] = saida_custo

    k+=1 # Atualiza o contador