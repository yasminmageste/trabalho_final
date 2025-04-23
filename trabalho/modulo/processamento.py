import cv2 #manipula imagens_roupas
import mediapipe as mp #detecta as partes do corpo
import numpy as np



def extrair_dados_da_imagem(imagem):
    #recebe uma imagem (formato BGR do OpenCV) e retorna um dicionário com medidas extraídas dela
    mp_pose = mp.solutions.pose
    medidas = {}
    with mp_pose.Pose(static_image_mode=True) as pose:
        img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB) #Converte a imagem de BGR (formato padrão do OpenCV) para RGB (formato exigido pelo MediaPipe)
        resultado = pose.process(img_rgb) #Processa a imagem

    if resultado.pose_landmarks: #verifica se deu certo
        # ALTURA TOTAL (CABEÇA AO TORNOZELO)
        topo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]  # ou .LEFT_EYE
        tornozelo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        altura_total = abs(tornozelo.y - topo.y)
        medidas['altura_total_normalizada'] = altura_total
        # DISTANCIA_OMBROS
        l_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        dx = (l_shoulder.x - r_shoulder.x)
        dy = (l_shoulder.y - r_shoulder.y)
        distancia = np.sqrt(dx**2 + dy**2)
        medidas['largura_ombros_normalizada'] = distancia
        #PROPORÇÃO TRONCO E PERNA
        ombro = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        quadril = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        tornozelo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        altura_tronco = abs(ombro.y - quadril.y)
        altura_pernas = abs(quadril.y - tornozelo.y)
        proporcao_tronco_pernas = altura_tronco / altura_pernas
        medidas['proporcao_tronco_pernas'] = proporcao_tronco_pernas
    else:
        print('nao deu certo os ombros')

    # Tom de pele (média da cor de uma área central do rosto)
    h, w, _ = imagem.shape #h= altura, w=altura, _=canal de cores(ignora)
    # Região estimada do rosto — pode ser ajustada dependendo da imagem
    rosto = imagem[int(h * 0.3):int(h * 0.5), int(w * 0.4):int(w * 0.6)]
    if rosto.size > 0:
        tom_medio = np.mean(rosto, axis=(0, 1))  # BGR
        medidas['tom_de_pele_bgr'] = tom_medio
    else:
        print('não deu certo a cara')
        return None

    return medidas, resultado

def visualizar_resultados(imagem, resultado, tom_de_pele):
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

    # Retângulo do rosto (onde foi tirado o tom de pele)
    h, w, _ = imagem.shape
    x1, y1 = int(w * 0.4), int(h * 0.3)
    x2, y2 = int(w * 0.6), int(h * 0.5)
    cv2.rectangle(imagem_copy, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Caixinha com a cor média do tom de pele
    tom = tuple([int(c) for c in tom_de_pele])
    cv2.rectangle(imagem_copy, (10, 10), (110, 110), tom, -1)
    cv2.putText(imagem_copy, "Tom de pele", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Mostra a imagem
    cv2.imshow("Visualizacao", imagem_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
