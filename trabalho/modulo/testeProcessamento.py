import os
from tkinter import filedialog
from tkinter import Tk
import cv2
from processamento import extrair_dados_da_imagem, visualizar_resultados

if __name__ == "__main__":
    # Cria a janela para escolher o arquivo
    Tk().withdraw()  # Oculta a janela principal do tkinter
    caminho = filedialog.askopenfilename(title="Selecione a imagem",
                                                   filetypes=[("Arquivos de imagem", "*.jpg;*.jpeg;*.png")])

    if not os.path.exists(caminho):
        print(f"Erro: imagem '{caminho}' não encontrada na pasta {os.getcwd()}")
    else:
        imagem = cv2.imread(caminho)
        if imagem is None:
            print("Erro ao carregar a imagem (talvez o arquivo esteja corrompido ou não seja uma imagem válida).")
        else:
            dados = extrair_dados_da_imagem(imagem)
            print("Medidas extraídas:")
            print(dados)

            medidas, resultado = extrair_dados_da_imagem(imagem)

            print("Medidas extraídas:")
            for k, v in medidas.items():
                print(f'{k}:{v}')

            if 'tom_de_pele_bgr' in medidas:
                visualizar_resultados(imagem, resultado, medidas['tom_de_pele_bgr'])