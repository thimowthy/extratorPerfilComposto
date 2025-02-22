import os
import pickle
import shutil

from tqdm import tqdm
from ImageClustering import ImageClustering

def criarPastasClusters(clusteringPath: ImageClustering, clusterFolder: str, inputFolder: str):
    
    clusterFolder = clusterFolder #os.path.join(os.getcwd(), "AGRUPAMENTO")
    inputFolder = inputFolder #os.path.join(os.getcwd(), "RECORTES")

    folders = os.listdir(inputFolder)

    all_images = []

    for folder_name in folders:
        images_path = os.path.join(inputFolder, folder_name, "partes")
        all_images.extend([os.path.join(images_path, img) for img in os.listdir(images_path)])


    with open(clusteringPath, 'rb') as file:
        
        clustering : ImageClustering = pickle.load(file)
        
        for i, cluster in tqdm(enumerate(clustering.clusters, start=1), desc="Agrupando Clusters em Pastas...", total=len(clustering.clusters)):
            
            folderPath = os.path.join(clusterFolder, str(cluster.name) if cluster.name else str(cluster.id))
            os.makedirs(folderPath, exist_ok=True)
            
            for image_path in all_images:
                if image_path in cluster.members:
                    
                    filename = os.path.basename(image_path)
                    second_level_folder = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
                    new_path = os.path.join(folderPath, f"{second_level_folder}_{filename}")
                    
                    shutil.copy(image_path, new_path)
