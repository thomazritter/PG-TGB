# Image Editor - Trabalho GB (Processamento Gráfico)

Este projeto é um editor gráfico desenvolvido como parte do Trabalho de Grau B na disciplina de Processamento Gráfico. Ele implementa funcionalidades inspiradas nos stories do Instagram, permitindo aos usuários editar imagens e vídeos por meio de filtros, stickers e outras manipulações gráficas.

## Funcionalidades Principais

### 1. **Carregar Imagem**

- Os usuários podem carregar uma imagem do computador para edição.
- **Local no código**: Função `load_new_image(file_path)`.

### 2. **Salvar Imagem**

- A imagem editada pode ser salva em formato PNG com os filtros e stickers aplicados.
- **Local no código**: Função `save_image(image, file_path)`.

### 3. **Capturar Foto**

- Permite capturar uma imagem diretamente da webcam.
- **Local no código**: Função `capture_photo()`.

### 4. **Aplicação de Filtros**

- O editor suporta os seguintes filtros:
  - Original
  - RGB
  - Sepia
  - Cinza
  - Negativo
  - Desfoque
  - Detecção de Bordas
  - Realce de Cores
  - Sharpen
  - Preto e Branco com Alto Contraste
- **Local no código**: Função `apply_filter(filter, image)`.

### 5. **Manipulação de Imagens**

- Rotação em incrementos de 90° (esquerda ou direita).
- Espelhamento horizontal e vertical.
- **Local no código**: Função `update_image(image, flipx, flipy, rotate, filter)`.

### 6. **Stickers**

- Os usuários podem adicionar stickers predefinidos à imagem:
  - Cada sticker é redimensionado e posicionado com base no clique do usuário.
  - **Local no código**: Função `add_sticker(x, y)`.

### 7. **Interface Gráfica**

- Desenvolvida com **PySimpleGUI** para uma experiência amigável e intuitiva.
- Permite interações por botões, clique para posicionamento de stickers e seleção de filtros.
- **Local no código**: Seção `layout` e `while True` (loop principal).

## Requisitos

- **Python 3.8+**
- Bibliotecas necessárias:
  - `opencv-python`
  - `numpy`
  - `PySimpleGUI`

Para instalar as dependências, execute:

```bash
pip install opencv-python numpy PySimpleGUI
```
