import os
import pandas as pd
import cv2
import numpy as np

def recomendar_roupas(dicionario):
    base_dir = os.path.dirname(__file__)
    caminho_csv = os.path.abspath(os.path.join(base_dir, '../../trabalho/data/catalogo_roupas.csv'))

    print("\nLendo CSV em:", caminho_csv)
    catalogo = pd.read_csv(caminho_csv)

    # Mostra o conteúdo lido
    catalogo.columns = catalogo.columns.str.strip().str.lower()
    print(catalogo.head())

    # Regras de recomendação
    #print("📦 Dicionário recebido:", dicionario)

    # Cria cópia do catálogo
    roupas_filtradas = catalogo.copy()

    # Regras de recomendação
    classificacao = dicionario.get('Classificação', '').lower()
    subtom = dicionario.get('Subtom', '').lower()

    if "baixo contraste escuro" in classificacao:
        roupas_filtradas = roupas_filtradas[
            roupas_filtradas['contraste'].str.contains("baixo contraste escuro", case=False)]
    elif "baixo contraste claro" in classificacao:
        roupas_filtradas = roupas_filtradas[
            roupas_filtradas['contraste'].str.contains("baixo contraste claro", case=False)]
    else:
        if "quente" in subtom:
            roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("quente", case=False)]
        elif "frio" in subtom:
            roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("frio|inverno", case=False)]
        elif "neutro" in subtom:
            roupas_filtradas = roupas_filtradas[roupas_filtradas['estação'].str.contains("neutro", case=False)]
        else:
            print("⚠️ Nenhuma condição atendida, mantendo todas as roupas.")
            roupas_filtradas = catalogo.copy()

    # DEBUG: Mostra valores únicos das colunas de filtragem
    print("\n🧪 Valores únicos de 'contraste':", catalogo['contraste'].unique())
    print("🧪 Valores únicos de 'estação':", catalogo['estação'].unique())

    # Converte string "[146 28 63]" para lista [146, 28, 63]
    roupas_filtradas["cor bgr"] = roupas_filtradas["cor bgr"].apply(lambda x: list(map(int, x.strip("[]").split())))

    # Filtra por tipo, se necessário
    roupas_filtradas = roupas_filtradas[roupas_filtradas["tipo"] == "camisa"]

    # Resultado final
    if not roupas_filtradas.empty:
        print("\n👕 ROUPAS FILTRADAS:")
        print(roupas_filtradas)

        cores_bgr = []

        for _, row in roupas_filtradas.iterrows():
            if isinstance(row['cor bgr'], list) and len(row['cor bgr']) == 3:
                cor = tuple(row['cor bgr'])
                cores_bgr.append(cor)
            if len(cores_bgr) == 20:
                break

        # === Exibir em grid ===
        quadrado = 100
        espaco = 20
        colunas = 5
        linhas = 4

        largura_total = colunas * (quadrado + espaco) + espaco
        altura_total = linhas * (quadrado + espaco) + espaco

        painel = np.full((altura_total, largura_total, 3), 255, dtype=np.uint8)

        for i, cor in enumerate(cores_bgr):
            linha = i // colunas
            coluna = i % colunas
            x = espaco + coluna * (quadrado + espaco)
            y = espaco + linha * (quadrado + espaco)
            cv2.rectangle(painel, (x, y), (x + quadrado, y + quadrado), color=cor, thickness=-1)

        cv2.imshow("20 Cores", painel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("⚠️ Nenhuma roupa recomendada.")
