#------------------------------HEADER------------------------------#
#Projeto computacional em que implementamos o método simplex.
'''AUTHORS:
Daniel Reis, Matheus Alves, Mariane Santana.
'''

import numpy as np
#------------------------------HEADER------------------------------#

#Função que lê da entrada as informações relevantes para o PL.
def entrada():
    #Lê o tamanho da matriz de restrições
    linhas = int(input("Informe o tamanho da matriz.\nLinhas: "))
    colunas = int(input("\nColunas: "))
    #Cria uma matriz (A)mxn de zeros
    A = np.zeros(linhas*colunas, dtype=float).reshape(linhas, colunas)

    #Atribui às entradas de A os valores do input do usuário.
    for i in range(linhas):
        A[i] = input().split()

    #Atribui às entradas de b (vetor de recursos) os valores do input do usuário.
    recursos = [float(b) for b in input("Informe o vetor dos recursos ("+str(linhas)+" entradas separadas por um espaço): ").split(sep = " ")]

    #Atribui às entradas de c (vetor de custos) os valores do input do usuário.
    custos = [float(c) for c in input("Informe o vetor dos custos ("+str(colunas)+" entradas separadas por um espaço, incluindo zeros para as variáveis de folga): ").split(sep = " ")]

    #Printa os valores lidos para confirmar se tudo correu bem.
    print('A = {}\n'.format(A))
    print('b = {}\n'.format(recursos))
    print('c = {}\n'.format(custos))
    return A, recursos, custos, linhas, colunas


#Função que particiona o PL.
def part(A,custos, linhas, colunas):
    # Define partições e os custos básicos e não-básicos
    particao_basica = A[:,(linhas-colunas):].copy()
    particao_nbasica = A[:,:linhas].copy()
    custos_basicos = custos[(linhas-colunas):].copy()
    custos_nbasicos = custos[:linhas].copy()
    return particao_basica, particao_nbasica, custos_basicos, custos_nbasicos

# Procura uma solução básica factível, Xb.
def solucao_basica(particao_basica, recursos):
    xb = np.linalg.solve(particao_basica, recursos)
    if(min(xb) < 0 or min(xb) == None):
        return None
    else:
        return xb
    
#Função que calcula o valor da função objetivo considerando apenas a parte básica.
def objetivo(custos_basicos, xb):
    valor_atual = np.dot(np.transpose(custos_basicos), xb)
    return valor_atual

#Função que calcula os custos relativos (e retorna o quê exatamente? Só os custos?) Ver tbm se a entrada recebe os parametros certos
def relativos(particao_basica, particao_nbasica, custos_basicos, custos_nbasicos, linhas, colunas):
    lambda_simplex = np.linalg.solve(np.transpose(particao_basica), custos_basicos)
    custos_relativos = [(custos_nbasicos[j] - np.dot(np.transpose(lambda_simplex), particao_nbasica[:,j])) for j in range(colunas-linhas)]
    
    return custos_relativos

#Funcao que performa o teste de otimalidade (Essa é tão simples que fico na duvida se cabe implementar direto no main()).
def otima(custos_relativos):
    custo_relativo_min = min(custos_relativos)
    indice_entrada_base = custos_relativos.index(custo_relativo_min)
    if(custo_relativo_min >= 0):
        return (-1)
    else:
        return indice_entrada_base
    
def indice_saida(xb, particao_basica, particao_nbasica, indice_entrada_base):
    direcao_simplex = np.linalg.solve(particao_basica, particao_nbasica[:,indice_entrada_base])
    for i in list((direcao_simplex <= 0)):
        if i == False:
            passo = xb/direcao_simplex
            epsilon = min(passo)
            indice_saida_base = np.where(passo == epsilon)[0][0]
            return indice_saida_base
    return (-1)

def troca_base(particao_basica, particao_nbasica, custos_basicos, custos_nbasicos, indice_entrada_base, indice_saida_base):
    # Realiza troca das colunas da base
    entrada_base = particao_nbasica[:,indice_entrada_base].copy()
    saida_base = particao_basica[:,indice_saida_base].copy()
    particao_basica[:,indice_saida_base] = entrada_base
    particao_nbasica[:,indice_entrada_base] = saida_base

    # Realiza troca dos custos básicos
    entrada_custo = custos_nbasicos[indice_entrada_base]
    saida_custo = custos_basicos[indice_saida_base]
    custos_basicos[indice_saida_base] = entrada_custo
    custos_nbasicos[indice_entrada_base] = saida_custo

    return particao_basica, particao_nbasica, custos_basicos, custos_nbasicos

def main():
    
    A, recursos, custos, linhas, colunas = entrada()
    particao_basica, particao_nbasica, custos_basicos, custos_nbasicos = part(A,custos, linhas, colunas)
    xb = solucao_basica(particao_basica, recursos) #Verificar se precisamos iterar sobre todas as possiveis formas de particionar.
    if(min(xb) == None):
        print('Solução básica factível não encontrada.')
        # exit(0) #Termina a execução do programa.

    custos_relativos = relativos(particao_basica, particao_nbasica, custos_basicos, custos_nbasicos, linhas, colunas)
    while(otima(custos_relativos) != -1):
        indice_entrada_base = otima(custos_relativos)
        indice_saida_base = indice_saida(xb, particao_basica, particao_nbasica, indice_entrada_base) #direcao de passo
        if indice_saida_base == (-1): #Se direcao é negativa ... O problema é ilimitado.
            print("O problema não tem solução ótima finita.")
            # exit(0) #break;
        else:
            #Funcao que faz a troca (lembrar de informar quais variaveis estao em cada particao na iteracao atual).
            particao_basica, particao_nbasica, custos_basicos, custos_nbasicos = troca_base(particao_basica,
                                                                                                particao_nbasica,
                                                                                                custos_basicos,
                                                                                                custos_nbasicos,
                                                                                                indice_entrada_base,
                                                                                                indice_saida_base)
            #refaz custo relativo
            custos_relativos = relativos(particao_basica, particao_nbasica, custos_basicos, custos_nbasicos, linhas, colunas)

    if(otima(custos_relativos)):
        valor_atual = objetivo(custos_basicos, xb)

        print("Solução atual é ótima.")
        print("Valor ótimo da solução: ", valor_atual)
        print("Variáveis ótimas: ", xb)
        # exit(0)