import cv2 # Manipula imagens_roupas
import mediapipe as mp # Detecta as partes do corpo
import numpy as np # Calcula distâncias


def extrair_dados_da_imagem(imagem):
    # Recebe uma imagem (formato BGR do OpenCV) e retorna um dicionário com medidas extraídas dela
    mp_pose = mp.solutions.pose
    medidas = {}
    with mp_pose.Pose(static_image_mode=True) as pose:
        img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB) # Converte a imagem de BGR (formato padrão do OpenCV) para RGB (formato exigido pelo MediaPipe)
        resultado = pose.process(img_rgb) # Processa a imagem

    if resultado.pose_landmarks:
        # Altura total (cabeça ao tonozelo)
        topo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]  # ou .LEFT_EYE
        tornozelo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        altura_total = round(abs(tornozelo.y - topo.y), 2)
        medidas['altura_total'] = altura_total
        # Disância entre os ombros
        l_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        dx = (l_shoulder.x - r_shoulder.x)
        dy = (l_shoulder.y - r_shoulder.y)
        distancia = round(np.sqrt(dx**2 + dy**2), 2)
        medidas['largura_ombros'] = distancia
        # Proporção tronco e perna
        ombro = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        quadril = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        tornozelo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        altura_tronco = abs(ombro.y - quadril.y)
        altura_pernas = abs(quadril.y - tornozelo.y)
        proporcao_tronco_pernas = round(altura_tronco / altura_pernas, 2)
        medidas['proporção'] = proporcao_tronco_pernas
    else:
        print('nao deu certo os ombros')

    h, w, _ = imagem.shape
    # Usa o ponto do nariz como centro da área de amostragem
    nariz = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    cx, cy = int(nariz.x * w), int(nariz.y * h)
    # Define o tamanho da área ao redor do nariz (ex: 40x40 pixels)
    offset = 20
    x1, y1 = max(cx - offset, 0), max(cy - offset, 0)
    x2, y2 = min(cx + offset, w), min(cy + offset, h)

    rosto = imagem[y1:y2, x1:x2]
    if rosto.size > 0:
        tom_medio = np.mean(rosto, axis=(0, 1))  # BGR
        medidas['tom_de_pele'] = tom_medio.astype(int)
    else:
        print('Não foi possível extrair o tom de pele')

    # Tom de cabelo (acima do nariz)
    cx, cy = int(nariz.x * w), int(nariz.y * h)
    offset_cabelo = 20
    # Subir mais um pouco para garantir que estamos acima da testa
    y1_cabelo = max(cy - 3 * offset_cabelo, 0)
    y2_cabelo = max(cy - offset_cabelo, 0)
    x1_cabelo = max(cx - offset_cabelo, 0)
    x2_cabelo = min(cx + offset_cabelo, w)

    cabelo = imagem[y1_cabelo:y2_cabelo, x1_cabelo:x2_cabelo]
    if cabelo.size > 0:
        tom_cabelo = np.mean(cabelo, axis=(0, 1))  # BGR
        medidas['tom_de_cabelo'] = tom_cabelo.astype(int)
    else:
        print('Não foi possível extrair o tom do cabelo')

    return medidas, resultado

def visualizar_resultados(imagem, resultado, tom_de_pele, tom_de_cabelo):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    imagem_copy = imagem.copy()

    # Desenha os pontos do corpo
    if resultado.pose_landmarks:
        mp_drawing.draw_landmarks(
            imagem_copy,
            resultado.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
        )

        # Retângulo do rosto baseado no nariz
        h, w, _ = imagem.shape
        nariz = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        cx, cy = int(nariz.x * w), int(nariz.y * h)
        offset = 20
        x1, y1 = max(cx - offset, 0), max(cy - offset, 0)
        x2, y2 = min(cx + offset, w), min(cy + offset, h)
        cv2.rectangle(imagem_copy, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Retângulo do cabelo acima do nariz
        y1_cb = max(cy - 3 * offset, 0)
        y2_cb = max(cy - offset, 0)
        x1_cb = max(cx - offset, 0)
        x2_cb = min(cx + offset, w)
        cv2.rectangle(imagem_copy, (x1_cb, y1_cb), (x2_cb, y2_cb), (0, 255, 255), 2)

    # Caixinha com a cor média do tom de pele
    tom_pele = tuple([int(c) for c in tom_de_pele])
    cv2.rectangle(imagem_copy, (10, 10), (110, 110), tom_pele, -1)
    cv2.putText(imagem_copy, "Tom de pele", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Caixinha com a cor média do tom de cabelo logo abaixo
    tom_cabelo = tuple([int(c) for c in tom_de_cabelo])
    cv2.rectangle(imagem_copy, (10, 150), (110, 250), tom_cabelo, -1)
    cv2.putText(imagem_copy, "Tom de cabelo", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Mostra a imagem
    cv2.imshow("Visualizacao", imagem_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
