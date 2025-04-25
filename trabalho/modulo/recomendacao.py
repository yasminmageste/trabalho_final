import os
import pandas as pd
from processamento import extrair_dados_da_imagem


def recomendar_roupas(medidas):
    caminho_base = os.path.dirname(os.path.abspath(__file__))
    caminho_csv = os.path.join(caminho_base, '..', 'data', 'catalogo_roupas.csv')

    print("\nLendo CSV em:", caminho_csv)
    catalogo = pd.read_csv(caminho_csv)

    # Mostra o conteúdo lido
    catalogo.columns = catalogo.columns.str.strip().str.lower()
    print(catalogo.head())

    # Exemplo de dados do usuário (simulados)
    roupas_filtradas = catalogo.copy()

    # Regras de recomendação
    if medidas.get('largura_ombros', 0) > 0:
        print('oi')
        roupas_filtradas = roupas_filtradas[roupas_filtradas['tipo'].str.contains("blusa", case=False)]

    if isinstance(medidas.get('tom_de_pele'), (list, tuple)) and len(medidas['tom_de_pele']) >= 3:
        if medidas['tom_de_pele'][2] < 100:
            roupas_filtradas = roupas_filtradas[roupas_filtradas['cor'].str.contains("preto", case=False, na=False)]

    return roupas_filtradas.head(3)  # Top 3 sugestões
