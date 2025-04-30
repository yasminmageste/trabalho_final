import cv2 #manipula imagens_roupas
import mediapipe as mp #detecta as partes do corpo
import numpy as np

def extrair_dados_da_imagem(imagem):
    #recebe uma imagem (formato BGR do OpenCV) e retorna um dicionário com medidas extraídas dela
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    medidas = {}

    # Converter para RGB uma única vez
    img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    # Processar pose e face juntos (melhor desempenho)
    with mp_pose.Pose(static_image_mode=True) as pose, mp_face_mesh.FaceMesh(
             static_image_mode=True,
             max_num_faces=1,
             refine_landmarks=True,
             min_detection_confidence=0.5
         ) as face_mesh:

        resultado = pose.process(img_rgb)
        resultado_face = face_mesh.process(img_rgb)

    # CORPO -------------------------------------------------------------------------------------
    if resultado.pose_landmarks:
        # ALTURA TOTAL (CABEÇA AO TORNOZELO)
        topo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]  # ou .LEFT_EYE
        tornozelo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        altura_total = round(abs(tornozelo.y - topo.y), 2)
        medidas['altura_total'] = altura_total

        # DISTANCIA_OMBROS
        l_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        dx = (l_shoulder.x - r_shoulder.x)
        dy = (l_shoulder.y - r_shoulder.y)
        distancia = round(np.sqrt(dx ** 2 + dy ** 2), 2)
        medidas['largura_ombros'] = distancia

        # PROPORÇÃO TRONCO E PERNA
        ombro = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        quadril = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        tornozelo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        altura_tronco = abs(ombro.y - quadril.y)
        altura_pernas = abs(quadril.y - tornozelo.y)
        proporcao_tronco_pernas = round(altura_tronco / altura_pernas, 2)
        medidas['proporção'] = proporcao_tronco_pernas
    else:
        print('Não foi possível detectar os ombros')

    cv2.imshow("Imagem de Entrada", imagem)
    
    # ROSTO -----------------------------------------------------------------
    h, w, _ = imagem.shape
    # Usa o ponto do nariz como centro da área de amostragem
    nariz = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    cx, cy = int(nariz.x * w), int(nariz.y * h)
    
    # Define o tamanho da área ao redor do nariz
    offset = 15
    x1, y1 = max(cx - offset, 0), max(cy - offset, 0)
    x2, y2 = min(cx + offset, w), min(cy + offset, h)

    rosto = imagem[y1:y2, x1:x2]
    if rosto.size > 0:
        tom_medio = np.mean(rosto, axis=(0, 1))  # BGR
    else:
        print('Não foi possível extrair o tom de pele')

    # Definir ROI (zoom 3x ao redor do nariz)
    zoom_factor = 3  # Ajuste conforme necessário
    roi_size = 150  # Tamanho da região ampliada (pixels)
    x1 = max(cx - roi_size // 2, 0)
    y1 = max(cy - roi_size // 2, 0)
    x2 = min(cx + roi_size // 2, w)
    y2 = min(cy + roi_size // 2, h)

    roi = imagem[y1:y2, x1:x2]

    # Ampliar a ROI (zoom no rosto)
    if roi.size > 0:
        roi_ampliada = cv2.resize(roi, (roi_size * zoom_factor, roi_size * zoom_factor))

        # Aplicar Face Mesh na ROI ampliada
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
        ) as face_mesh:
            roi_rgb = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2RGB)
            resultado_face = face_mesh.process(roi_rgb)

            if resultado_face.multi_face_landmarks:
                face_landmarks = resultado_face.multi_face_landmarks[0]

                # Usar ponto 4 (ponta do nariz) e dimensões da ROI
                ponto_nariz = face_landmarks.landmark[4]
                h_roi, w_roi, _ = roi_ampliada.shape
                x_nose, y_nose = int(ponto_nariz.x * w_roi), int(ponto_nariz.y * h_roi)

                offset = 8  # Reduzido para área mais precisa
                x1 = max(x_nose - offset, 0)
                y1 = max(y_nose - offset, 0)
                x2 = min(x_nose + offset, w_roi)  # Corrigido para w_roi
                y2 = min(y_nose + offset, h_roi)  # Corrigido para h_roi

                regiao_pele = roi_ampliada[y1:y2, x1:x2]

                if regiao_pele.size > 0:
                    # Aplicar filtro HSV
                    regiao_hsv = cv2.cvtColor(regiao_pele, cv2.COLOR_BGR2HSV)
                    mask_pele = cv2.inRange(regiao_hsv, (0, 30, 60), (25, 150, 255))
                    regiao_filtrada = cv2.bitwise_and(regiao_pele, regiao_pele, mask=mask_pele)

                    # Calcular média apenas nos pixels de pele
                    tom_pele = cv2.mean(regiao_filtrada, mask=mask_pele)[:3]
                    medidas['tom_de_pele'] = np.array(tom_pele).astype(int)

                    # Debug visual
                    debug_img = roi_ampliada.copy()
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow("Nariz Analisado", debug_img)
            else:
                print("Landmarks faciais não detectados.")
    # CABELO ---------------------------------------------------------------------------------
    if face_landmarks:
        try:
            h_roi, w_roi, _ = roi_ampliada.shape

            # Encontrar ponto mais baixo do rosto (queixo)
            ponto_queixo = max(
                [(int(lm.x * w_roi), int(lm.y * h_roi)) for lm in face_landmarks.landmark[152:155]],
                key=lambda p: p[1]
            )

            # Converter para HSV
            hsv = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2HSV)

            # Faixa para tons loiros (amarelos claros a dourados)
            loiro_min = np.array([15, 40, 160])  # H, S, V
            loiro_max = np.array([45, 180, 255])

            mask_loiro = cv2.inRange(hsv, loiro_min, loiro_max)

            # Máscara para destacar áreas escuras (potencial cabelo)
            gray = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2GRAY)
            _, mask_escura = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

            # Máscara do rosto baseada nos landmarks
            mask_rosto = np.zeros_like(gray)
            pontos_rosto = np.array([
                (int(lm.x * w_roi), int(lm.y * h_roi)) for lm in face_landmarks.landmark[:468]
            ])
            cv2.fillConvexPoly(mask_rosto, pontos_rosto, 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask_rosto = cv2.dilate(mask_rosto, kernel)

            # Subtrair o rosto da máscara escura
            mask_cabelo_total = cv2.bitwise_or(mask_loiro, mask_escura)
            mask_cabelo = cv2.subtract(mask_cabelo_total, mask_rosto)

            # Restringir a região superior (acima do queixo)
            mask_regiao = np.zeros_like(gray)
            cv2.rectangle(mask_regiao, (0, 0), (w_roi, ponto_queixo[1] - 10), 255, -1)
            mask_cabelo = cv2.bitwise_and(mask_cabelo, mask_regiao)

            # Encontrar contornos
            contornos, _ = cv2.findContours(mask_cabelo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                maior_contorno = max(contornos, key=cv2.contourArea)
                mascara_final = np.zeros_like(gray)
                cv2.drawContours(mascara_final, [maior_contorno], -1, 255, -1)

                media_hsv = cv2.mean(hsv, mask=mascara_final)
                if media_hsv[1] < 25 or media_hsv[2] > 245:
                    # Saturação baixa ou brilho muito alto -> provável fundo
                    print("Descartado: cor muito clara (fundo ou pele)")
                    medidas['pouco_cabelo'] = True
                else:
                    media_bgr = cv2.mean(roi_ampliada, mask=mascara_final)[:3]
                    medidas['tom_de_cabelo'] = np.array(media_bgr).astype(int)

                # Extrair pixels do cabelo
                pixels_cabelo = cv2.bitwise_and(roi_ampliada, roi_ampliada, mask=mascara_final)
                media_cabelo = cv2.mean(pixels_cabelo, mask=mascara_final)[:3]
                medidas['tom_de_cabelo'] = np.array(media_cabelo).astype(int)
                medidas['pouco_cabelo'] = False

                # DEBUG VISUAL
                debug_img = roi_ampliada.copy()
                cv2.drawContours(debug_img, [maior_contorno], -1, (0, 255, 0), 2)
                cv2.line(debug_img, (0, ponto_queixo[1] - 10), (w_roi, ponto_queixo[1] - 10), (0, 0, 255), 2)
                cv2.imshow("Região Analisada - Cabelo", debug_img)
                area_cabelo = cv2.countNonZero(mascara_final)
                limite_area_minima = 500  # Ajuste conforme seus testes (valor empírico)

                if area_cabelo < limite_area_minima:
                    print("Área de cabelo muito pequena – possível calvície.")
                    medidas['pouco_cabelo'] = True
                    medidas['tom_de_cabelo'] = None
                else:
                    # Extrair pixels do cabelo
                    pixels_cabelo = cv2.bitwise_and(roi_ampliada, roi_ampliada, mask=mascara_final)
                    media_cabelo = cv2.mean(pixels_cabelo, mask=mascara_final)[:3]
                    medidas['tom_de_cabelo'] = np.array(media_cabelo).astype(int)
                    medidas['pouco_cabelo'] = False

            else:
                medidas['pouco_cabelo'] = True
                print("Nenhum contorno de cabelo detectado.")

        except Exception as e:
            print(f"Erro na análise de cabelo: {str(e)}")
            medidas['pouco_cabelo'] = True
    # CONTRASTE ---------------------------------------------------------------------------------
    # Adicionando após obter os tons de pele e cabelo:
    def bgr_to_gray_scale_0_10(bgr):
        gray = int(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])
        escala = np.clip(round(gray / 255 * 10), 0, 10)
        return escala

    # Obter escala de cinza dos tons extraídos
    if 'tom_de_pele' in medidas and 'tom_de_cabelo' in medidas and medidas['tom_de_cabelo'] is not None:
        escala_pele = bgr_to_gray_scale_0_10(medidas['tom_de_pele'])
        escala_cabelo = bgr_to_gray_scale_0_10(medidas['tom_de_cabelo'])

        # Para simplificação, estimar os olhos próximos ao tom médio entre pele e cabelo
        escala_olhos = round((escala_pele + escala_cabelo) / 2)

        # Encontrar tons extremos
        tons = [escala_pele, escala_cabelo, escala_olhos]
        tom_min = min(tons)
        tom_max = max(tons)
        intervalo = tom_max - tom_min

        # Classificação de contraste
        if intervalo <= 3:
            contraste = "baixo contraste"
        elif intervalo <= 6:
            contraste = "contraste médio"
        else:
            contraste = "alto contraste"

        medidas["Tom de pele (escala 0-10)"] = escala_pele
        medidas["Tom de cabelo (escala 0-10)"] = escala_cabelo
        medidas["Tom dos olhos (estimado)"] = escala_olhos
        medidas["Intervalo de contraste"] = intervalo
        medidas["Classificação"] = contraste
    else:
        print("Dados insuficientes para calcular contraste.")

    return medidas, resultado


def visualizar_resultados(imagem, resultado, tom_de_pele, pouco_cabelo, tom_de_cabelo=None):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Primeiro processamos a imagem ORIGINAL (sem bordas)
    imagem_landmarks = imagem.copy()

    # Landmarks CORPORAIS (exatamente como na sua versão original)
    if resultado.pose_landmarks:
        mp_drawing.draw_landmarks(
            imagem_landmarks,
            resultado.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    # Criar painel de resultados
    painel_resultados = np.full((imagem.shape[0], 300, 3), 240, dtype=np.uint8)

    # Cabeçalho
    cv2.putText(painel_resultados, "RESULTADOS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Tom de pele
    cv2.putText(painel_resultados, "Tom de pele:", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(painel_resultados, (20, 90), (120, 170),
                  tuple([int(c) for c in tom_de_pele]), -1)
    cv2.putText(painel_resultados, f"RGB: {list(tom_de_pele)}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Tom de cabelo
    y_cabelo = 230
    cv2.putText(painel_resultados, "Tom de cabelo:", (20, y_cabelo),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if not pouco_cabelo and tom_de_cabelo is not None:
        cv2.rectangle(painel_resultados, (20, y_cabelo + 30), (120, y_cabelo + 110),
                      tuple([int(c) for c in tom_de_cabelo]), -1)
        cv2.putText(painel_resultados, f"RGB: {list(tom_de_cabelo)}", (20, y_cabelo + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        cv2.putText(painel_resultados, "Nao detectado", (20, y_cabelo + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Combinar as imagens HORIZONTALMENTE (não usando copyMakeBorder)
    imagem_final = np.hstack((imagem_landmarks, painel_resultados))

    # Exibição
    cv2.namedWindow("Analise Corporal", cv2.WINDOW_NORMAL)
    cv2.imshow("Analise Corporal", imagem_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
