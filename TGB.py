import os
import cv2 as cv
import numpy as np
import PySimpleGUI as sg
from io import BytesIO

# Trabalho GB - Processamento Gráfico
# Alunos: Carlos Souza e Thomaz Justo
# Este script implementa um editor gráfico com funcionalidade de aplicação de filtros,
# adição de stickers, manipulação de imagens e captura de vídeo inspirado nos stories do Instagram.

# Variáveis Globais
flipx = False
flipy = False
rotate = 0
current_filter = 0
selected_sticker = None
img = None
bio = None
stickers = []  # Lista de stickers (posição x, y, caminho do arquivo)

# Função para aplicar filtros na imagem
def apply_filter(filter, image):
    if image is None:
        sg.popup_error("Nenhuma imagem carregada. Por favor, abra ou capture uma imagem antes de aplicar filtros.")
        return image
    image = image.copy()
    if filter == 1:  # Filtro RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    elif filter == 2:  # Filtro Sepia
        sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        image = cv.transform(image, sepia_matrix)
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif filter == 3:  # Filtro Cinza
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif filter == 4:  # Filtro Negativo
        image = cv.bitwise_not(image)
    elif filter == 5:  # Desfoque
        image = cv.GaussianBlur(image, (15, 15), 0)
    elif filter == 6:  # Detecção de Bordas
        edges = cv.Canny(image, 100, 200)
        image = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    elif filter == 7:  # Realce de Cores
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv.add(hsv[:, :, 1], 50)
        image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    elif filter == 8:  # Filtro Sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        image = cv.filter2D(image, -1, kernel)
    elif filter == 9:  # Preto e Branco com Alto Contraste
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
        image = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    return image

# Função para salvar a imagem com filtros e stickers
def save_image(image, file_path):
    if image is None:
        sg.popup_error("Nenhuma imagem carregada. Por favor, abra ou capture uma imagem antes de salvar.")
        return

    # Aplicar o filtro atual na imagem antes de salvar
    filtered_image = apply_filter(current_filter, image)

    # Adicionar stickers à imagem
    h, w = filtered_image.shape[:2]
    for sticker in stickers:
        x, y, s = sticker
        overlay = cv.imread(s, cv.IMREAD_UNCHANGED)

        if overlay is not None:
            overlay = cv.resize(overlay, (100, 100))
            y1, y2 = max(0, y), min(y + overlay.shape[0], h)
            x1, x2 = max(0, x), min(x + overlay.shape[1], w)

            alpha_s = overlay[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                filtered_image[y1:y2, x1:x2, c] = (alpha_s[:y2-y1, :x2-x1] * overlay[:y2-y1, :x2-x1, c] +
                                                   alpha_l[:y2-y1, :x2-x1] * filtered_image[y1:y2, x1:x2, c])

    # Salvar a imagem com o filtro e os stickers aplicados
    cv.imwrite(file_path, filtered_image)

# Função para atualizar a imagem na interface
def update_image(image, flipx, flipy, rotate, filter):
    if image is None:
        sg.popup_error("Nenhuma imagem carregada. Por favor, abra ou capture uma imagem antes de aplicar ações.")
        return

    global bio
    img = image.copy()
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Rotação
    if rotate != 0:
        rotation_matrix = cv.getRotationMatrix2D(center=center, angle=rotate, scale=1)
        img = cv.warpAffine(img, rotation_matrix, (w, h))

    # Espelhamento
    if flipx:
        img = cv.flip(img, 1)
    if flipy:
        img = cv.flip(img, 0)

    # Aplicar filtro
    img = apply_filter(filter, img)

    # Adicionar stickers
    for sticker in stickers:
        x, y, s = sticker
        overlay = cv.imread(s, cv.IMREAD_UNCHANGED)

        if overlay is not None:
            overlay = cv.resize(overlay, (100, 100))
            y1, y2 = max(0, y), min(y + overlay.shape[0], h)
            x1, x2 = max(0, x), min(x + overlay.shape[1], w)

            alpha_s = overlay[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                img[y1:y2, x1:x2, c] = (alpha_s[:y2-y1, :x2-x1] * overlay[:y2-y1, :x2-x1, c] +
                                        alpha_l[:y2-y1, :x2-x1] * img[y1:y2, x1:x2, c])

    # Atualizar interface
    _, buf = cv.imencode('.png', img)
    bio = BytesIO(buf)
    window['-GRAPH-'].erase()  # Limpa a área antes de desenhar
    window['-GRAPH-'].draw_image(data=bio.getvalue(), location=(0, 600))

# Função para carregar uma nova imagem
def load_new_image(file_path):
    global img, stickers, flipx, flipy, rotate, current_filter

    # Resetar as variáveis
    stickers = []
    flipx = False
    flipy = False
    rotate = 0
    current_filter = 0

    # Carregar a nova imagem
    img = cv.imread(file_path)
    if img is not None:
        window['-GRAPH-'].erase()  # Limpar o Graph antes de desenhar
        update_image(img, flipx, flipy, rotate, current_filter)
    else:
        sg.popup_error("Erro ao carregar a imagem.")

# Função para capturar foto
def capture_photo():
    global img, stickers, flipx, flipy, rotate, current_filter

    # Resetar as variáveis
    stickers = []
    flipx = False
    flipy = False
    rotate = 0
    current_filter = 0

    # Iniciar webcam
    video_capture = cv.VideoCapture(0)
    if not video_capture.isOpened():
        sg.popup_error("Erro ao acessar a webcam.")
        return

    sg.popup("Pressione [ESPACO] para capturar ou [ESC] para cancelar.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            sg.popup_error("Erro ao capturar vídeo.")
            break

        cv.imshow("Captura de Foto", frame)
        key = cv.waitKey(1) & 0xFF

        if key == 32:  # ESPAÇO para capturar
            img = frame
            cv.destroyAllWindows()
            break
        elif key == 27:  # ESC para cancelar
            cv.destroyAllWindows()
            break

    video_capture.release()

    if img is not None:
        window['-GRAPH-'].erase()
        update_image(img, flipx, flipy, rotate, current_filter)

# Função para adicionar sticker com o clique como centro
def add_sticker(x, y):
    if img is None:
        sg.popup_error("Nenhuma imagem carregada. Por favor, abra ou capture uma imagem antes de adicionar stickers.")
        return

    global stickers

    h, w = img.shape[:2]
    sticker_size = 100  # Tamanho fixo do sticker (100x100)

    # Ajustar as coordenadas para que o clique seja o centro do sticker
    x_center = int(x - sticker_size / 2)
    y_center = int(h - y - sticker_size / 2)  # Inverter Y para coordenada da imagem

    # Verificar se o sticker está dentro dos limites da imagem
    if 0 <= x_center <= w - sticker_size and 0 <= y_center <= h - sticker_size:
        stickers.append((x_center, y_center, selected_sticker))  # Adiciona o sticker
        update_image(img, flipx, flipy, rotate, current_filter)  # Atualiza a imagem
    else:
        sg.popup_error("Clique dentro da área da imagem para adicionar o sticker.")  # Exibe mensagem de erro

# Layout da interface
sg.theme('DarkAmber')

# Botões de filtros
filter_buttons = [
    sg.Button('Original', key='-FILTER-0-', size=(10, 1)),
    sg.Button('RGB', key='-FILTER-1-', size=(10, 1)),
    sg.Button('Sepia', key='-FILTER-2-', size=(10, 1)),
    sg.Button('Cinza', key='-FILTER-3-', size=(10, 1)),
    sg.Button('Negativo', key='-FILTER-4-', size=(10, 1)),
    sg.Button('Desfoque', key='-FILTER-5-', size=(10, 1)),
    sg.Button('Bordas', key='-FILTER-6-', size=(10, 1)),
    sg.Button('Realce', key='-FILTER-7-', size=(10, 1)),
    sg.Button('Sharpen', key='-FILTER-8-', size=(10, 1)),
    sg.Button('Preto/Branco', key='-FILTER-9-', size=(10, 1))
]

# Carregar Stickers
sticker_files = [f for f in os.listdir('./assets/stickers') if f.endswith('.png')]
sticker_buttons = [
    sg.Button('', image_filename=f'./assets/stickers/{sticker}', key=f'-STICKER-{i}-', size=(5, 2))
    for i, sticker in enumerate(sticker_files)
]

layout = [
    [sg.Text('Editor de Imagens - Carlos & Thomaz', size=(40, 1), justification='center', font=('Helvetica', 16))],
    [sg.Graph(canvas_size=(800, 600), graph_bottom_left=(0, 0), graph_top_right=(800, 600),
              enable_events=True, key='-GRAPH-')],
    [sg.Button('Abrir Imagem', key='-OPEN-', size=(12, 1)), sg.Button('Salvar Imagem', key='-SAVE-', size=(12, 1))],
    [sg.Button('Capturar Foto', key='-CAPTURE-', size=(16, 1))],
    [sg.Button('Girar Esquerda', key='-ROTATEL-', size=(12, 1)), sg.Button('Girar Direita', key='-ROTATER-', size=(12, 1))],
    [sg.Button('Espelhar Horizontal', key='-FLIPX-', size=(12, 1)), sg.Button('Espelhar Vertical', key='-FLIPY-', size=(12, 1))],
    [sg.Frame('Filtros', layout=[filter_buttons], title_location='n', font='Any 12')],
    [sg.Button('Remover Stickers', key='-CLEAR-STICKERS-', size=(16, 1))],
    [sg.Frame('Stickers', layout=[sticker_buttons], title_location='n', font='Any 12')]
]

window = sg.Window('Editor de Imagens', layout, finalize=True)

# Loop principal
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    elif event == '-OPEN-':
        file_path = sg.popup_get_file('Abrir Imagem', no_window=True)
        if file_path:
            load_new_image(file_path)

    elif event == '-GRAPH-' and selected_sticker:
        x, y = values['-GRAPH-']
        add_sticker(x, y)

    elif event == '-SAVE-':
        if img is not None:
            file_path = sg.popup_get_file('Salvar Imagem', save_as=True, no_window=True, file_types=(("PNG Files", "*.png"),))
            if file_path:
                save_image(img, file_path)
        else:
            sg.popup_error("Nenhuma imagem carregada para salvar.")

    elif event == '-CAPTURE-':
        capture_photo()

    elif event == '-ROTATEL-':
        if img is not None:
            rotate -= 90
            update_image(img, flipx, flipy, rotate, current_filter)

    elif event == '-ROTATER-':
        if img is not None:
            rotate += 90
            update_image(img, flipx, flipy, rotate, current_filter)

    elif event == '-FLIPX-':
        if img is not None:
            flipx = not flipx
            update_image(img, flipx, flipy, rotate, current_filter)

    elif event == '-FLIPY-':
        if img is not None:
            flipy = not flipy
            update_image(img, flipx, flipy, rotate, current_filter)

    # Selecionar filtro
    elif event.startswith('-FILTER-'):
        filter_index = int(event.split('-')[-2])
        current_filter = filter_index
        update_image(img, flipx, flipy, rotate, current_filter)

    elif event.startswith('-STICKER'):
        idx = int(event.split('-')[-2])
        selected_sticker = f'./assets/stickers/{sticker_files[idx]}'

    elif event == '-CLEAR-STICKERS-':
        stickers = []
        if img is not None:
            update_image(img, flipx, flipy, rotate, current_filter)

window.close()
