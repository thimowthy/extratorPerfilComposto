import cv2
from Image import Image
from PIL import Image as PillowImage

def compare_images(image1: Image, image2: Image):
    img1, img2 = image1.image_matrix, image2.image_matrix
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    
    image1 = image1.crop_image(h, w)
    image2 = image2.crop_image(h, w)

    diff = cv2.absdiff(image1, image2)
    diff_channels = cv2.split(diff)

    num_pixels = image1.shape[0] * image1.shape[1] * image1.shape[2]
    total_diff = sum(cv2.sumElems(channel)[0] for channel in diff_channels)
    similarity = 1 - (total_diff / (255 * num_pixels))

    return similarity


class ImageCluster:


    def __init__(self, image: Image, id: int):
        
        def getClusterName():
            img_path = image.path
            img = PillowImage.open(img_path)
            img.show()
                
            nome_cluster = input("\nDigite um nome para o cluster: ").upper().strip()
            return nome_cluster
        
        self.image = image
        self.id = id
        self.name = getClusterName()
        self.image_path = image.path
        self.color_reference = image.most_common_color
        self.density_reference = image.black_pixel_density_value
        self.members = set([image.path])

    def add_member(self, path: str):
        self.members.add(path)
    
    def clear_members(self):
        self.members = set()


class ImageClustering:

    def __init__(self, *clusters: ImageCluster, height: int = 10, density_threshold: float = 0.1, similarity_threshold : float = 0.85):
        self.clusters = list(clusters)
        self.h = height
        self.density_threshold = density_threshold
        self.similarity_threshold = similarity_threshold
        

    def clean_data(self):
        for cluster in self.clusters:
            cluster.clear_members()


    def add_cluster(self, cluster: ImageCluster):
        self.clusters.append(cluster)

    
    def get_cluster(self, img: Image):
        for cluster in self.clusters:
            if img.path in cluster.members:
                return cluster

    
    def cluster_image(self, img: Image):
        
        possible_clusters = []
        
        for cluster in self.clusters:
            density_difference = abs(cluster.density_reference - img.black_pixel_density_value)
            similarity = compare_images(cluster.image, img)
            
            if cluster.color_reference == img.most_common_color:
                if density_difference <= self.density_threshold and similarity >= self.similarity_threshold:
                    possible_clusters.append(cluster)

        if possible_clusters:
            chosen_cluster = max(possible_clusters, key=lambda cluster: compare_images(cluster.image, img))
            chosen_cluster.add_member(img.path)
        else:
            id = len(self.clusters)
            self.add_cluster(ImageCluster(img, id))
