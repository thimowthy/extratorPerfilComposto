import os
import pickle
import pandas as pd
from tqdm import tqdm
from Image import Image
from ImageClustering import *
from PIL import Image as PImage

RECORTES_FOLDER = r"C:\Users\55799\Desktop\PIBIC\Projeto-PIBIC\RECORTES"

def calculate_heights(image_folder_path, topo, constante):
    
    cum_height = topo
    depths = []

    for image in sorted(os.listdir(image_folder_path), key=lambda x:int(x.replace("img_", "").replace(".png", ""))):
        img_path = os.path.join(image_folder_path, image)
        img = PImage.open(img_path)
        img_height = img.size[1]
        depths.append(cum_height)
        cum_height += img_height*constante
    else:
        depths.append(cum_height)
    
    return depths


def get_labels(image_folder_path):
    
    file_path = os.path.join(os.getcwd(), "clustering_final.pkl")
    
    with open(file_path, 'rb') as file:    
        clustering : ImageClustering = pickle.load(file)
        labels = []
        
        for image in sorted(os.listdir(image_folder_path), key=lambda x:int(x.replace("img_", "").replace(".png", ""))):
            img_path = os.path.join(image_folder_path, image)
            img = Image(img_path, None)
            cluster = clustering.get_cluster(img)
            labels.append(cluster.name or cluster.id)
        
        return labels
            

def generate_csv(output_path, depths, labels):
    min_depth = 0.0
    max_depth = max(depths)
    depth_range = [round(min_depth + i * 0.1, 2) for i in range(int((max_depth - min_depth) / 0.1) + 1)]
    lithologies = [None]*int(max_depth*10 + 1)
    
    last = int(depths[0]*10)
    
    # print(list(zip(depths[1:], labels))[-1])
    
    for topo, label in zip(depths[1:], labels):
        topo = round(topo*10)
        for i in range(last, topo):
            lithologies[i] = label
        last = topo
        
    # print(last, int(depths[-1]*10) + 1)
    
    for i in range(last, int(depths[-1]*10) + 1):
        lithologies[i] = labels[-1]
            
    df = pd.DataFrame({"TDEP": depth_range, "LITOLOGIA": lithologies})
    df.to_csv(output_path, index=False)
    
  
def createDepthLabelTable(inputFolder: str):
    for folder in tqdm(os.listdir(inputFolder), desc="Criando arquivos CSV"):
        image_folder_path = os.path.join(inputFolder, folder, "partes")
        constante, topo = open(os.path.join(inputFolder, folder, "constante_topo.txt"), 'r').readline().split(" ")
        
        depths = calculate_heights(image_folder_path, float(topo), float(constante))
        labels = get_labels(image_folder_path)
        # print(folder)
        # pprint(list(zip(depths, labels)))
        output_csv = os.path.join(inputFolder, folder, f"lithology_depths_{folder}.csv")
        generate_csv(output_csv, depths, labels)

if __name__=="__main__":
    createDepthLabelTable(RECORTES_FOLDER)