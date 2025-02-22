import colorsys
import os
import shutil
from PIL import Image
from collections import Counter

from tqdm import tqdm


def is_dark_color(color, threshold=0.2):
    r, g, b = color
    _, _, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return v < threshold

def crop_image_based_on_color(image_path, subFolder):
    def mode_of_pixels(pixels):
        """Finds the mode (most common color) of a list of pixels."""
        return Counter(pixels).most_common(1)[0][0]
    
    img_folder = os.path.join(subFolder, "partes")
    
    shutil.rmtree(img_folder, ignore_errors=True)

    os.makedirs(img_folder)

    image = Image.open(image_path)
    pixels = image.load()
    width, height = image.size
    
    KERNEL = (width, 7) # 5 altura

    first_block_pixels = [ pixels[x, y] for x in range(min(KERNEL[0], width)) for y in range(min(KERNEL[1], height)) ]
    starting_color = mode_of_pixels(first_block_pixels)

    c = 1
    last_crop = 0, 0
    
    for y in range(height):

        line_width = 2
        
        if pixels[0, y] != starting_color:
            row_colors = Counter([pixels[x, y] for x in range(width)])
            modal_color, count = row_colors.most_common(1)[0]
            mode_predominance =  count / width
            is_dark = is_dark_color(modal_color)
            
            
            if starting_color not in row_colors or row_colors[starting_color] < 3:
                if y + KERNEL[1] < height:
                    next_block_pixels = [ pixels[x, y + dy] for x in range(min(KERNEL[0], width)) for dy in range(KERNEL[1]) ]
                    next_block_mode = mode_of_pixels(next_block_pixels)
                    
                    # if folder == "1-BRSA-551-SE_perfil_composto":
                    #     print(starting_color, next_block_mode, c)
                        
                    if next_block_mode != starting_color:
                        img_path = os.path.join(img_folder, f"img_{c}.png")
                        
                        # if mode_predominance >= 0.95 and not is_dark:
                        #     line_width = 1
                        
                        line_width = 1
                        
                        try:
                            cropped_image = image.crop((last_crop[0], last_crop[1], width, y+line_width))
                            cropped_image.save(img_path)
                        except:
                            line_width = 2
                            cropped_image = image.crop((last_crop[0], last_crop[1], width, y+line_width))
                            cropped_image.save(img_path)
                            
                        starting_color = next_block_mode
                        last_crop = 0, y + line_width
                        c += 1

    img_path = os.path.join(img_folder, f"img_{c}.png")                                            
    cropped_image = image.crop((last_crop[0], last_crop[1], width, height))
    cropped_image.save(img_path)


def extractImages(inputFolder: str):
    for folder in tqdm(os.listdir(inputFolder), desc="Extraindo Imagens"):
        subFolder = os.path.join(inputFolder, folder)
        image_path = os.path.join(subFolder, "imagem_recortada.png")
        crop_image_based_on_color(image_path, subFolder)

if __name__=="__main__":
    RECORTES_FOLDER = r"C:\Users\55799\Desktop\PIBIC\Projeto-PIBIC\RECORTES"
    extractImages(RECORTES_FOLDER)