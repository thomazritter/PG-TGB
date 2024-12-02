import os, base64
import numpy as np
import cv2 as cv
import PySimpleGUI as sg
from PIL import ImageGrab
from io import BytesIO

# Trabalho GB - Processamento Gráfico
# Alunos: Carlos Souza e Thomaz Justo
# Este script implementa um editor gráfico com funcionalidade de aplicação de filtros,
# adição de stickers, manipulação de imagens e captura de vídeo inspirado nos stories do Instagram.

# ---------------------- FUNÇÕES ---------------------- #
def update_elements():
    # Captura os botões da interface gráfica
    capture_button = window['-CAPTURE-']
    record_button = window['-RECORD-']
    flipx_button = window['-FLIPX-']
    flipy_button = window['-FLIPY-']
    filter_buttons = [window[f'-FILTER{i}-'] for i in range(11)]
    sticker_buttons = [window[f'-STICKER{i}-'] for i in range(11)]

    # Atualiza o estado do botão de captura
    capture_button.update(disabled=not record)
    
    # Atualiza a cor do botão de gravação
    record_button.update(button_color='red' if record else 'white')
    
    # Atualiza a cor do botão de flip horizontal
    flipx_button.update(button_color='gold' if flipx else 'white')
    
    # Atualiza a cor do botão de flip vertical
    flipy_button.update(button_color='gold' if flipy else 'white')
    
    # Atualiza a cor dos botões de filtro
    for i, button in enumerate(filter_buttons):
        button.update(button_color='gold' if current_filter == i else 'black')
    
    # Atualiza a cor dos botões de sticker
    sticker_images = [
        sticker_image_0, sticker_image_1, sticker_image_2, sticker_image_3,
        sticker_image_4, sticker_image_5, sticker_image_6, sticker_image_7,
        sticker_image_8, sticker_image_9, sticker_image_10
    ]
    for i, button in enumerate(sticker_buttons):
        button.update(button_color='gold' if current_sticker == sticker_images[i] else 'black')

def apply_filter(filter, image):
    # Cria uma cópia da imagem original para não modificar a original
    image = image.copy()

    # Filtro 1: Aumenta a intensidade do canal vermelho
    if filter == 1:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float64)
        image = cv.transform(image, np.matrix([[1, 0, 0], [1.10104433,  0, -0.00901975], [0, 0, 1]]))
        image[np.where(image > 255)] = 255
        image = np.array(image, dtype=np.uint8)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Filtro 2: Aumenta a intensidade do canal verde
    elif filter == 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float64)
        image = cv.transform(image, np.matrix([[0, 0.90822864, 0.008192], [0, 1, 0], [0, 0, 1]]))
        image[np.where(image > 255)] = 255
        image = np.array(image, dtype=np.uint8)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Filtro 3: Aumenta a intensidade do canal azul
    elif filter == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float64)
        image = cv.transform(image, np.matrix([[1, 0, 0], [0, 1, 0], [-0.15773032,  1.19465634, 0]]))
        image[np.where(image > 255)] = 255
        image = np.array(image, dtype=np.uint8)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Filtro 4: Inverte as cores da imagem (efeito negativo)
    elif filter == 4:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i, j, 0] = image[i, j, 0] ^ 255
                image[i, j, 1] = image[i, j, 1] ^ 255
                image[i, j, 2] = image[i, j, 2] ^ 255

    # Filtro 5: Converte a imagem para preto e branco com limiarização
    elif filter == 5:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i, j] = 0 if image[i, j] < 120 else 255
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        
    # Filtro 6: Aplica detecção de bordas usando o algoritmo Canny
    elif filter == 6:
        image = cv.Canny(image, 50, 150)

    # Filtro 7: Converte a imagem para tons de cinza
    elif filter == 7:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                avg = (
                    image[i, j, 0] * 0.333
                    + image[i, j, 1] * 0.333
                    + image[i, j, 2] * 0.3333
                )
                image[i, j, 0] = avg
                image[i, j, 1] = avg
                image[i, j, 2] = avg

    # Filtro 8: Aplica dilatação à imagem
    elif filter == 8:
        kernel = np.ones((3, 3), np.uint8)
        image = cv.dilate(image, kernel, iterations=2)

    # Filtro 9: Aplica erosão à imagem
    elif filter == 9:
        kernel = np.ones((3, 3), np.uint8)
        image = cv.erode(image, kernel, iterations=2)

    # Filtro 10: Aplica um efeito sépia à imagem
    elif filter == 10:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float64)
        image = cv.transform(image, np.matrix([[0.393, 0.769, 0.189],
                                               [0.349, 0.686, 0.168],
                                               [0.272, 0.534, 0.131]]))
        image[np.where(image > 255)] = 255
        image = np.array(image, dtype=np.uint8)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    return image

def update_image(image, blur, contrast, brightness, sharpen, emboss, contour, flipx, flipy, rotate, filter):
    global img, bio
    img = image.copy()  # Cria uma cópia da imagem original para não modificá-la diretamente
    h, w = img.shape[:2]  # Obtém a altura e a largura da imagem
    c = (w / 2, h / 2)  # Define o centro da imagem para rotações

    # Aplica desfoque à imagem
    blur = int(blur) + 1
    img = cv.blur(img, (blur, blur))

    # Aplica rotação à imagem
    rotation_matrix = cv.getRotationMatrix2D(center=c, angle=rotate, scale=1)
    img = cv.warpAffine(src=img, M=rotation_matrix, dsize=(w, h))

    # Ajusta o contraste e o brilho da imagem
    alpha = 1.0 + (contrast * 0.02)
    beta = brightness
    img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Aplica nitidez à imagem
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv.filter2D(img, -1, kernel)

    # Aplica efeito de relevo à imagem
    if emboss:
        kernel = np.array([[-1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1]])
        img = cv.filter2D(img, -1, kernel)

    # Aplica contorno à imagem
    if contour:
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        img = cv.filter2D(img, -1, kernel)

    # Aplica flip horizontal à imagem
    if flipx:
        img = cv.flip(img, 1)

    # Aplica flip vertical à imagem
    if flipy:
        img = cv.flip(img, 0)

    # Aplica o filtro selecionado à imagem
    img = apply_filter(filter, img)

    # Codifica a imagem em formato PNG e armazena em um buffer
    _, buf = cv.imencode('.png', img)
    bio = BytesIO(buf)

    # Atualiza a imagem na interface gráfica
    window['-IMAGE-'].update(data=bio.getvalue())

# DIRECTORIES #

work_dir = os.path.dirname(os.path.abspath(__file__))
icon_dir = os.path.join(work_dir, 'assets', 'icons')
sticker_dir = os.path.join(work_dir, 'assets', 'stickers')
# ---------------------------------------------------- #

# ASSETS #
flipx_icon = os.path.join(icon_dir, 'flipx.png')
flipy_icon = os.path.join(icon_dir, 'flipy.png')
rotater_icon = os.path.join(icon_dir, 'rotater.png')
rotatel_icon = os.path.join(icon_dir, 'rotatel.png')
# ---------------------------------------------------- #

# VARIABLES #
applied = False
graph_save = False
record = False
capture = False
flipx = False
flipy = False
rotate = 0
current_filter = 0
current_sticker = None
# ---------------------------------------------------- #

# IMAGES #
image_path = ''
image = cv.imread(image_path)

sample_image_path = os.path.join(work_dir, 'nature.png')
sample_image = cv.imread(sample_image_path)
sample_image = cv.resize(sample_image, dsize=(64,64), interpolation=cv.INTER_CUBIC)

filter_image_0 = cv.imencode('.png', apply_filter(0, sample_image))[1].tobytes()
filter_image_1 = cv.imencode('.png', apply_filter(1, sample_image))[1].tobytes()
filter_image_2 = cv.imencode('.png', apply_filter(2, sample_image))[1].tobytes()
filter_image_3 = cv.imencode('.png', apply_filter(3, sample_image))[1].tobytes()
filter_image_4 = cv.imencode('.png', apply_filter(4, sample_image))[1].tobytes()
filter_image_5 = cv.imencode('.png', apply_filter(5, sample_image))[1].tobytes()
filter_image_6 = cv.imencode('.png', apply_filter(6, sample_image))[1].tobytes()
filter_image_7 = cv.imencode('.png', apply_filter(7, sample_image))[1].tobytes()
filter_image_8 = cv.imencode('.png', apply_filter(8, sample_image))[1].tobytes()
filter_image_9 = cv.imencode('.png', apply_filter(9, sample_image))[1].tobytes()
filter_image_10 = cv.imencode('.png', apply_filter(10, sample_image))[1].tobytes()

sticker0_image_path = os.path.join(sticker_dir, 'cancel.png')
sticker1_image_path = os.path.join(sticker_dir, 'star.png')
sticker2_image_path = os.path.join(sticker_dir, 'sun.png')
sticker3_image_path = os.path.join(sticker_dir, 'moon.png')
sticker4_image_path = os.path.join(sticker_dir, 'rainbow.png')
sticker5_image_path = os.path.join(sticker_dir, 'leaf.png')
sticker6_image_path = os.path.join(sticker_dir, 'water.png')
sticker7_image_path = os.path.join(sticker_dir, 'fire.png')
sticker8_image_path = os.path.join(sticker_dir, 'dog.png')
sticker9_image_path = os.path.join(sticker_dir, 'cat.png')
sticker10_image_path = os.path.join(sticker_dir, 'fox.png')

with open(sticker0_image_path, 'rb') as f:
    sticker_image_0 = base64.b64encode(f.read())
with open(sticker1_image_path, 'rb') as f:
    sticker_image_1 = base64.b64encode(f.read())
with open(sticker2_image_path, 'rb') as f:
    sticker_image_2 = base64.b64encode(f.read())
with open(sticker3_image_path, 'rb') as f:
    sticker_image_3 = base64.b64encode(f.read())
with open(sticker4_image_path, 'rb') as f:
    sticker_image_4 = base64.b64encode(f.read())
with open(sticker5_image_path, 'rb') as f:
    sticker_image_5 = base64.b64encode(f.read())
with open(sticker6_image_path, 'rb') as f:
    sticker_image_6 = base64.b64encode(f.read())
with open(sticker7_image_path, 'rb') as f:
    sticker_image_7 = base64.b64encode(f.read())
with open(sticker8_image_path, 'rb') as f:
    sticker_image_8 = base64.b64encode(f.read())
with open(sticker9_image_path, 'rb') as f:
    sticker_image_9 = base64.b64encode(f.read())
with open(sticker10_image_path, 'rb') as f:
    sticker_image_10 = base64.b64encode(f.read())
# ---------------------------------------------------- #

# UI #
sg.theme('Black')

editor_col = sg.Column([
    [sg.Frame('Blur', pad=(0, 5), layout=[[sg.Slider(range=(0,100), orientation='h', s=(30,10), key='-BLUR-', enable_events=True)]]),],
    [sg.Frame('Contrast', pad=(0, 5), layout=[[sg.Slider(range=(0,100), orientation='h', s=(30,10), key='-CONTRAST-', enable_events=True)]]),],
    [sg.Frame('Brightness', pad=(0, 5), layout=[[sg.Slider(range=(0,100), orientation='h', s=(30,10), key='-BRIGHTNESS-', enable_events=True)]]),],
    [
        sg.Frame('Flip', pad=((0,18), (5,5)), layout=[[
            sg.Frame('', pad=0, layout=[[sg.Button(image_source=flipx_icon, pad=0, image_subsample=16, key='-FLIPX-', mouseover_colors='gold')]]),
            sg.Frame('', pad=0, layout=[[sg.Button(image_source=flipy_icon, pad=0, image_subsample=16, key='-FLIPY-', mouseover_colors='gold')]]),
        ]]), 
        sg.Frame('Rotate', pad=((18,0), (5,5)), layout=[[
            sg.Frame('', pad=0, layout=[[sg.Button(image_source=rotatel_icon, pad=0, image_subsample=16, key='-ROTATEL-', mouseover_colors='gold')]]),
            sg.Frame('', pad=0, layout=[[sg.Button(image_source=rotater_icon, pad=0, image_subsample=16, key='-ROTATER-', mouseover_colors='gold')]]),
        ]]),
    ],
    [           
        sg.Checkbox('Emboss', key='-EMBOSS-', enable_events=True),
        sg.Checkbox('Contour', key='-CONTOUR-', enable_events=True),
        sg.Checkbox('Sharpen', key='-SHARPEN-', enable_events=True),
    ],
    [sg.Button('Open', pad=(0, (10,1)), s=(35,1), key='-OPEN-', enable_events=True)],
    [sg.Button('Record', pad=0, s=(35,1), key='-RECORD-', enable_events=True)],
    [sg.Button('Capture', pad=(0, (1,10)), s=(35,1), key='-CAPTURE-', enable_events=True, disabled_button_color='gray')],
])
image_col = sg.Column([
        [sg.Frame('', element_justification='center', expand_x=True, expand_y=True, layout=[[sg.Image(image_path, key='-IMAGE-')]])],
        [sg.Frame('Filters', layout=[[
            sg.Button(image_source=filter_image_0, enable_events=True, border_width=0, key='-FILTER0-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_1, enable_events=True, border_width=0, key='-FILTER1-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_2, enable_events=True, border_width=0, key='-FILTER2-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_3, enable_events=True, border_width=0, key='-FILTER3-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_4, enable_events=True, border_width=0, key='-FILTER4-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_5, enable_events=True, border_width=0, key='-FILTER5-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_6, enable_events=True, border_width=0, key='-FILTER6-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_7, enable_events=True, border_width=0, key='-FILTER7-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_8, enable_events=True, border_width=0, key='-FILTER8-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_9, enable_events=True, border_width=0, key='-FILTER9-', mouseover_colors='gold', button_color='black', pad=0),
            sg.Button(image_source=filter_image_10, enable_events=True, border_width=0, key='-FILTER10-', mouseover_colors='gold', button_color='black', pad=0),
        ]])]
    ], key='-IMGCOL-', visible=False)
stickers_row = [sg.Button('Stickers', pad=(5, 0), s=(16,1), key='-NEXT-', enable_events=True)]

editor_layout = [[editor_col, image_col],stickers_row]

graph_layout = [
    [sg.Frame('', element_justification='center', expand_x=True, expand_y=True, layout=[[sg.Graph((0,0),(0,0),(0,0),key='-IMGRAPH-',enable_events=True,change_submits=True)]])],
    [sg.Frame('Stickers', layout=[[
        sg.Button(image_source=sticker_image_0, enable_events=True, border_width=0, key='-STICKER0-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_1, enable_events=True, border_width=0, key='-STICKER1-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_2, enable_events=True, border_width=0, key='-STICKER2-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_3, enable_events=True, border_width=0, key='-STICKER3-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_4, enable_events=True, border_width=0, key='-STICKER4-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_5, enable_events=True, border_width=0, key='-STICKER5-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_6, enable_events=True, border_width=0, key='-STICKER6-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_7, enable_events=True, border_width=0, key='-STICKER7-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_8, enable_events=True, border_width=0, key='-STICKER8-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_9, enable_events=True, border_width=0, key='-STICKER9-', mouseover_colors='gold', button_color='black', pad=0),
        sg.Button(image_source=sticker_image_10, enable_events=True, border_width=0, key='-STICKER10-', mouseover_colors='gold', button_color='black', pad=0),
    ]])],
    [sg.Button('Discard', pad=(5, (10,0)), s=(16,1), key='-BACK-', enable_events=True)]
]

layout = [[sg.Column(editor_layout, key='-EDITOR-'), sg.Column(graph_layout, visible=False, key='-GRAPH-')], [sg.Button('Save', pad=(10, (5,10)), s=(16,1), key='-SAVE-')]]

window = sg.Window('paint 4d', layout)
graph = window['-IMGRAPH-']
video = cv.VideoCapture(0)

while True:
    event, values = window.read(timeout=0)
    
    if image is not None:
        window['-IMGCOL-'].update(visible=True)
        if event == '-FLIPX-':
            flipx = not flipx
    
        if event == '-FLIPY-':
            flipy = not flipy
    
        if event == '-ROTATER-':
            rotate -= 90
    
        if event == '-ROTATEL-':
            rotate += 90

        if event == '-FILTER0-':
            current_filter = 0
        if event == '-FILTER1-':
            current_filter = 1
        if event == '-FILTER2-':
            current_filter = 2
        if event == '-FILTER3-':
            current_filter = 3
        if event == '-FILTER4-':
            current_filter = 4
        if event == '-FILTER5-':
            current_filter = 5
        if event == '-FILTER6-':
            current_filter = 6
        if event == '-FILTER7-':
            current_filter = 7
        if event == '-FILTER8-':
            current_filter = 8
        if event == '-FILTER9-':
            current_filter = 9
        if event == '-FILTER10-':
            current_filter = 10

        if event == '-STICKER0-':
            current_sticker = sticker_image_0
        if event == '-STICKER1-':
            current_sticker = sticker_image_1
        if event == '-STICKER2-':
            current_sticker = sticker_image_2
        if event == '-STICKER3-':
            current_sticker = sticker_image_3
        if event == '-STICKER4-':
            current_sticker = sticker_image_4
        if event == '-STICKER5-':
            current_sticker = sticker_image_5
        if event == '-STICKER6-':
            current_sticker = sticker_image_6
        if event == '-STICKER7-':
            current_sticker = sticker_image_7
        if event == '-STICKER8-':
            current_sticker = sticker_image_8
        if event == '-STICKER9-':
            current_sticker = sticker_image_9
        if event == '-STICKER10-':
            current_sticker = sticker_image_10

        if event == '-IMGRAPH-':
            x, y = values['-IMGRAPH-']
            if current_sticker:
                graph.draw_image(location=(x-32, y+32), data=current_sticker)

        if event == '-SAVE-':
            if graph_save:
                widget = graph.Widget
                box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())
                grab = ImageGrab.grab(bbox=box)
                save_path = sg.popup_get_file('Save', save_as=True, no_window=True) + '.png'
                grab.save(save_path)
            else:
                save_path = sg.popup_get_file('Save', save_as=True, no_window=True) + '.png'
                cv.imwrite(save_path, img)

        if event == '-NEXT-':
            graph_save = True
            window['-EDITOR-'].update(visible=False)
            window['-GRAPH-'].update(visible=True)
            graph.set_size(window['-IMAGE-'].get_size())
            graph.change_coordinates((0,0),window['-IMAGE-'].get_size())
            graph.draw_image(data=bio.getvalue(), location=(0, window['-IMAGE-'].get_size()[1]))
        
        if event == '-BACK-':
            graph_save = False
            graph.erase()
            current_sticker = None
            window['-EDITOR-'].update(visible=True)
            window['-GRAPH-'].update(visible=False)
        
        update_image(image, values['-BLUR-'], values['-CONTRAST-'], values['-BRIGHTNESS-'], values['-SHARPEN-'], values['-EMBOSS-'], values['-CONTOUR-'], flipx, flipy, rotate, current_filter)
    elif not record:
        window['-IMGCOL-'].update(visible=False)

    if record: 
        _, frame = video.read()
        width = int(frame.shape[1]*0.8)
        height = int(frame.shape[0]*0.8)
        frame = cv.resize(frame, (width, height), interpolation = cv.INTER_AREA)
        frame = cv.flip(frame, 1)
        framebytes = cv.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=framebytes)
        if capture:
            image_path = sg.popup_get_file('Save', save_as=True, no_window=True) + '.png'
            cv.imwrite(image_path, frame)
            image = cv.imread(image_path)
            window['-IMAGE-'].update(source=image_path)
            record = False
            capture = False
            
    if event == '-OPEN-':
        record = False
        image_path = sg.popup_get_file('Open', no_window=True)
        image = cv.imread(image_path)
        window['-IMAGE-'].update(source=image_path)

    if event == '-RECORD-':
        window['-IMGCOL-'].update(visible=True)
        record = True

    if event == '-CAPTURE-' and record:
        capture = True

    if event == sg.WIN_CLOSED:
        break
    update_elements()
video.release()
window.close()
# ---------------------------------------------------- #

