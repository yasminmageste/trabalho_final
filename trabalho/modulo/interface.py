import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from processamento import extrair_dados_da_imagem
from recomendacao import recomendar_roupas

def mostrar_interface():
    st.title("👗 Recomendador de Roupas com IA")

    arquivo = st.file_uploader("Envie uma imagem sua com uma roupa", type=['jpg', 'png'])
    if arquivo:
        imagem = Image.open(arquivo).convert('RGB')
        imagem_np = np.array(imagem)

        st.image(imagem_np, caption="Imagem enviada", use_column_width=True)

        dados = extrair_dados_da_imagem(imagem_np)
        st.write("Medidas extraídas:", dados)

    sugestoes = recomendar_roupas(dados)
    st.subheader("Sugestões de roupas:")
    for _, linha in sugestoes.iterrows():
        st.text(f"{linha['nome']} - {linha['cor']} - {linha['estilo']}")
        caminho = os.path.join("data", "imagens_roupas", linha["imagem"])
        st.image(caminho, width=200)
