import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np
import os
from statistics import mode, mean, median
from tqdm import tqdm


VALOR_APROX_DISTANCIA_5M = 58


def pdf_para_imagem(pdf_path, dpi=300, pasta_saida=None):
    """
    Converte as páginas de um PDF em imagens de alta resolução e as salva na pasta especificada.
    """
    doc = fitz.open(pdf_path)
    imagens = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        imagem = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        if pasta_saida:
            # Salva a imagem na pasta especificada
            caminho_imagem = os.path.join(pasta_saida, f"pagina_{page_num + 1}.png")
            imagem.save(caminho_imagem)
        
        imagens.append(imagem)
    return imagens


def remover_cinza_preto(imagem, pasta_saida=None):
    imagem_np = np.array(imagem)


    hsv = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2HSV)

    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([179, 128, 255])
    mask_cinza_preto = cv2.inRange(hsv, lower_bound, upper_bound)

    lower_red = np.array([0, 100, 100])  # Red hue range
    upper_red = np.array([10, 255, 255])  # Lower red range in HSV
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    mask = np.maximum(mask_cinza_preto, mask_red)
    branco = np.full_like(imagem_np, fill_value=255)

    imagem_sem_cinza_preto = np.where(mask[:, :, None], branco, imagem_np)
    imagem_sem_cinza_preto = Image.fromarray(imagem_sem_cinza_preto)

    if pasta_saida:
        caminho_imagem = os.path.join(pasta_saida, "imagem_sem_cinza_preto.png")
        imagem_sem_cinza_preto.save(caminho_imagem)

    return imagem_sem_cinza_preto


def obter_margem_e_contorno(imagem, pasta_saida=None):
    
    imagem_np = np.array(imagem)

    # Conversão para HSV e criação da máscara
    hsv = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2HSV)
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones((5, 1), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Nenhum objeto colorido encontrado.")
        return None, None

    contornos_ordenados = sorted(contours, key=cv2.contourArea, reverse=True)
    maior_contorno = contornos_ordenados[0]

    x, y, w, h = cv2.boundingRect(maior_contorno)

    largura_tolerancia = 20  # Tolerância para largura semelhante (10 pixels)
    altura_tolerancia = 200
    contorno_combinado = maior_contorno.copy()

    for contorno in contornos_ordenados[1:]:
        cx, cy, cw, ch = cv2.boundingRect(contorno)
        
        if abs(x - cx) <= largura_tolerancia and abs((cx + cw) - (x + w)) <= largura_tolerancia:
        # if (x <= cx <= (x + largura_tolerancia)) or (x <= cx + cw <= (x + w + largura_tolerancia)):
        # if abs(cw - w) <= largura_tolerancia:
            contorno_combinado = np.vstack((contorno_combinado, contorno))
    
    x, y, w, h = cv2.boundingRect(contorno_combinado)

    contorno_original = x, y, w, h
    
    x -= 10
    w += 10
    y -= 500
    h += 500

    x = max(x, 0)
    y = max(y, 0)
    w = min(w, imagem_np.shape[1] - x)
    h = min(h, imagem_np.shape[0] - y)

    contorno_margem = x, y, w, h
    
    if pasta_saida:
        caminho_imagem = os.path.join(pasta_saida, "contorno_e_margem.png")
        cv2.rectangle(imagem_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
        imagem_com_contorno = Image.fromarray(imagem_np)
        imagem_com_contorno.save(caminho_imagem)

    return (contorno_margem), (contorno_original)


def aplicar_contorno_e_margem(imagem, margem_e_contorno, pasta_saida=None):
        
    def recortar_imagem(imagem):
        imagem_array = np.array(imagem)
        
        porcentagem_preta = np.mean(imagem_array == 0, axis=0)
        colunas_quase_pretas = porcentagem_preta > 0.95
        
        if np.any(colunas_quase_pretas):
            ultima_coluna_quase_preta = np.where(colunas_quase_pretas)[0].max()
            if ultima_coluna_quase_preta <= imagem_array.shape[1] // 2:
                ultima_coluna_quase_preta = imagem_array.shape[1]
        else:
            ultima_coluna_quase_preta = imagem_array.shape[1]
        
        imagem_recortada = imagem_array[:, :ultima_coluna_quase_preta]
        
        return Image.fromarray(imagem_recortada)

    (mx, my, mw, mh), (x, y, w, h) = margem_e_contorno

    imagem_np = np.array(imagem)

    imagem_margem = imagem_np[my:my+mh, mx:mx+mw].copy()
    imagem_recortada = imagem_np[y:y+h, x:x+w].copy()
        
    #cv2.rectangle(imagem_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

    imagem_margem = Image.fromarray(imagem_margem)
    imagem_recortada = recortar_imagem(Image.fromarray(imagem_recortada))

    if pasta_saida:
        caminho_margem = os.path.join(pasta_saida, "imagem_margem.png")
        caminho_recortada = os.path.join(pasta_saida, "imagem_recortada.png")
        
        imagem_margem.save(caminho_margem)
        imagem_recortada.save(caminho_recortada)

    return imagem_margem, imagem_recortada


def ler_primeira_coluna_contar_brancos(imagem):

    if not isinstance(imagem, np.ndarray):
        imagem = np.array(imagem)

    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    primeira_coluna = imagem[:, 0]

    branco = 255

    contagens_brancos = []
    contagem_atual = 0
    encontrou_pixel_nao_branco = False

    for pixel in primeira_coluna:
        
        if pixel == branco:
            if encontrou_pixel_nao_branco:
                contagem_atual += 1
        else:
            if encontrou_pixel_nao_branco and contagem_atual > 0:
                contagens_brancos.append(contagem_atual)
                contagem_atual = 0
            encontrou_pixel_nao_branco = True

    if contagem_atual > 0:
        contagens_brancos.append(contagem_atual)

    contagens_brancos = [c for c in contagens_brancos if abs(c - VALOR_APROX_DISTANCIA_5M) < 15]
    
    return contagens_brancos


def ler_primeira_coluna_baixo_para_cima(imagem, num_pixels):

    if not isinstance(imagem, np.ndarray):
        imagem = np.array(imagem)

    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    primeira_coluna = imagem[:, 0]

    branco = 255

    contagem_pixels = 0
    sequencia_brancos = 0

    for pixel in reversed(primeira_coluna):
        if pixel == branco:
            sequencia_brancos += 1
            if sequencia_brancos == num_pixels:
                contagem_pixels -= num_pixels
                break
        else:
            sequencia_brancos = 0
        contagem_pixels += 1

    return contagem_pixels



def extrair_constante_topo(inputFolder: str, outputFolder: str):

    pdfs = os.listdir(inputFolder)

    pdfs_path = [os.path.join(inputFolder, pdf) for pdf in pdfs]


    for arquivo, pdf_path in tqdm(zip(pdfs, pdfs_path), desc="Processando PDFs", total=len(pdfs)):
    # for arquivo, pdf_path in zip(pdfs, pdfs_path):
        constantes_conversao = []

        pasta_saida = os.path.join(outputFolder, arquivo.split('.')[0])
        os.makedirs(pasta_saida, exist_ok=True)
        
        imagens = pdf_para_imagem(pdf_path, pasta_saida=pasta_saida)
        
        CONSTANTE_DE_AJUSTE = 2.0534 # Obtida a partir do poço 1-BRSA-510D-AL, cuja profundidade final marca exatamente 3370m (valor obtido 3370.000711220692)
        
        for i, imagem in enumerate(imagens):
            
            imagem_sem_linhas = remover_cinza_preto(imagem, pasta_saida=pasta_saida)
            margem_e_contorno = obter_margem_e_contorno(imagem_sem_linhas, pasta_saida=pasta_saida)
            
            if margem_e_contorno[0] is not None:
                
                imagem_margem, imagem_recortada = aplicar_contorno_e_margem(imagem, margem_e_contorno, pasta_saida=pasta_saida)
                #constantes_conversao.append(mode(ler_primeira_coluna_contar_brancos(imagem_margem)))
                distancias = ler_primeira_coluna_contar_brancos(imagem_margem)
                pixel_X_5m = mode(distancias) + CONSTANTE_DE_AJUSTE
                constante_conversao = 5/pixel_X_5m
                            
                comprimento_poco_total = ler_primeira_coluna_baixo_para_cima(imagem_margem, int(pixel_X_5m) + 10)*constante_conversao
                comprimento_poco = imagem_recortada.size[1]*constante_conversao
                topo = comprimento_poco_total - comprimento_poco
                
                # print(arquivo)
                # print(comprimento_poco_total)
                # print(topo, '\n')
                
                open(os.path.join(pasta_saida, "constante_topo.txt"), "w+").write(str(constante_conversao) + ' ' + str(topo))


if __name__ == "__main__":
    
    inputFolder = r"C:\Users\55799\Desktop\PIBIC\Projeto-PIBIC\PDF"
    outputFolder = r"C:\Users\55799\Desktop\PIBIC\Projeto-PIBIC\RECORTES"
    
    extrair_constante_topo(inputFolder, outputFolder)