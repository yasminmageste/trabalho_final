import os
import pandas as pd

def recomendar_roupas():
    caminho_base = os.path.dirname(os.path.abspath(__file__))
    caminho_csv = os.path.join(caminho_base, '..', 'data', 'catalogo_roupas.csv')

    print("Tentando ler CSV em:", caminho_csv)
    catalogo = pd.read_csv(caminho_csv)

    # Mostra o conteúdo lido
    catalogo.columns = catalogo.columns.str.strip().str.lower()

    print("\nPrimeiras linhas do DataFrame:")
    print(catalogo.head())

    # Exemplo de dados do usuário (simulados)
    dados_usuario = {
        'largura_ombros': 0.35,
        'tom_de_pele': (120, 100, 80)  # BGR
    }

    roupas_filtradas = catalogo.copy()

    # Regras de recomendação
    if dados_usuario['largura_ombros'] > 0.3:
        roupas_filtradas = roupas_filtradas[roupas_filtradas['tipo'].str.contains("blusa", case=False)]

    if dados_usuario['tom_de_pele'][2] < 100:
        roupas_filtradas = roupas_filtradas[roupas_filtradas['cor'].str.contains("preto", case=False)]

    return roupas_filtradas.head(3)  # Top 3 sugestões
