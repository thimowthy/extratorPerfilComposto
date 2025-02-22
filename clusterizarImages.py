import os
import cv2
import pickle
import colorsys
from tqdm import tqdm
from Image import Image
from ImageClustering import *
from collections import Counter


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


def find_minimum_height(image_list: list, set: set):
    
    smallest_height = float('inf')
    path = ""

    for image_path in image_list:
        img = cv2.imread(image_path)
        height, _, _ = img.shape

        if height < smallest_height and height not in set:
            smallest_height = height
            path = image_path

    return smallest_height, path


def find_mode_height(image_list: list, set: set):
    heights = []

    for image_path in image_list:
        img = cv2.imread(image_path)
        height, _, _ = img.shape
        heights.append(height)

    height_counter = Counter(heights)
    most_common_height, _ = height_counter.most_common(1)[0]

    return most_common_height


def cluster_images(image_paths: list, h: int, density_threshold: float = 0.1, similarity_threshold : float = 0.85, clustering: ImageClustering = None):

    if not clustering:
        path = next(iter(image_paths))
        first_image = Image(path, h)
        first_cluster = ImageCluster(first_image, 0)
        clustering = ImageClustering(first_cluster, height=h, density_threshold=density_threshold, similarity_threshold=similarity_threshold)    
    
    n = len(image_paths)
    i, c = 0, 0

    for i in range(n):
        path = image_paths[i] 
        img = Image(path, h)
        clustering.cluster_image(img)
        
        # possible_clusters = []

        # for cluster in clustering.clusters:
        #     density_difference = abs(cluster.density_reference - img.black_pixel_density_value)
        #     similarity = compare_images(cluster.image, img)
            
        #     if cluster.color_reference == img.most_common_color:
        #         if density_difference <= density_threshold and similarity >= similarity_threshold:
        #             possible_clusters.append(cluster)

        # if possible_clusters:
        #     chosen_cluster = max(possible_clusters, key=lambda cluster: compare_images(cluster.image, img))
        #     chosen_cluster.add_member(img.path)
        #     i += 1
        # else:
        #     clustering.add_cluster(ImageCluster(img))
        #     c += 1
        #     i += 1

    return clustering


def optimize_clustering(clustering: ImageClustering, threshold_members: int = 5, similarity_threshold: float=0.5) -> ImageClustering:
        
    sparse_clusters = [cluster for cluster in clustering.clusters
                       if len(cluster.members) <= threshold_members] 
    
    dense_clusters = [cluster for cluster in clustering.clusters
                       if len(cluster.members) > threshold_members] 

    for sparse in sparse_clusters:
        most_similar_cluster = max(dense_clusters, key=lambda cluster: compare_images(sparse.image, cluster.image))
        
        is_similar = compare_images(sparse.image, most_similar_cluster.image) >= similarity_threshold
        same_color = sparse.image.color_mode == most_similar_cluster.image.color_mode
        
        if is_similar and same_color:
            for image_path in sparse.members:
                most_similar_cluster.add_member(image_path)
        else:
            dense_clusters.append(sparse)

    optimized_clustering = ImageClustering(*dense_clusters, height=clustering.h)

    return optimized_clustering


def hsv(rgb: tuple):
    return colorsys.rgb_to_hsv(*rgb)
    #return math.sqrt(0.241*r + 0.691*g + .068*b )



def getClustering(imgsFolder: str, clusteringPath: str = None):

    if not clusteringPath:
        clusteringPath = os.path.join(os.getcwd(), "clustering_final.pkl")
    try:
        with open(clusteringPath, 'rb') as file:
            clustering : ImageClustering = pickle.load(file)
    except:
        clustering = None
        
    PATH = imgsFolder#os.path.join(os.getcwd(), "RECORTES")
    
    folders = os.listdir(PATH)

    all_images = []

    for folder_name in tqdm(folders, ncols=100, desc="Lendo Pastas das Imagens dos Poços..."):
        images_path = os.path.join(PATH, folder_name, "partes")
        images = sorted([os.path.join(images_path, img) for img in os.listdir(images_path)], key=lambda x:int(x.split(".")[0].split("_")[-1]))
        
        for img in images:
            all_images.append(img)

    h, _ = find_minimum_height(all_images, set())
    h  = find_mode_height(all_images, set())
    
    
    clustering = cluster_images(all_images, h, clustering=clustering)
    optimized_clustering = optimize_clustering(clustering)

    with open(clusteringPath, 'wb') as file:
        pickle.dump(optimized_clustering, file)

    return clusteringPath



# # plotly graph
# file_path = os.path.join(os.getcwd(), "clustering_teste.pkl")
# with open(file_path, 'rb') as file:
#     clustering : ImageClustering = pickle.load(file)

#     densities = [cluster.density_reference for cluster in clustering.clusters]
#     colors = [cluster.color_reference for cluster in clustering.clusters]
#     cluster_mapping = { cluster.image_path:i for i, cluster in enumerate(clustering.clusters) }
    
#     n = 5
#     images = [Image(img, clustering.h) for cluster in clustering.clusters
#               for img in cluster.members[:n]]
    
#     densities = [img.black_pixel_density_value for img in images]
#     colors = [img.most_common_color for img in images]
    

#     hover_data = [f"RGB: {cluster.color_reference} \n path:{cluster.image_path}"
#                   for cluster in clustering.clusters for _ in cluster.members[:n]]

#     category_colors = { cluster.image_path:cluster.color_reference for cluster in clustering.clusters }

#     del images
    
#     sorted_data = sorted(zip(colors, densities), key=lambda x: hsv(x[0]))

#     colors, densities = zip(*sorted_data)

#     color_to_index = {}
#     index = 0
    
#     numerical_colors = []
#     for color in colors:
#         if color not in color_to_index:
#             color_to_index[color] = index
#             index += 1
#         numerical_colors.append(color_to_index[color])


#     normalized_colors = np.array(colors) / 255.0
#     cores = [f'rgb{tuple(color)}' for color in normalized_colors]

#     fig = go.Figure(data=go.Scatter(x=list(range(len(densities))), 
#                                      y=densities, 
#                                      mode='markers',
#                                      marker=dict(color=cores,
#                                                  size=8,
#                                                  opacity=1),
#                                     hovertext=hover_data))


#     fig.update_layout(
#         title='Distribution \n Black Pixel Density x Color',
#         xaxis=dict(title='Index'),
#         yaxis=dict(title='Black Pixel Density'),
#         coloraxis_colorbar=dict(title='Numerical Color'),
#         width=800,
#         height=600
#     )
#     fig.show()

# matplot graph
# file_path = os.path.join(os.getcwd(), "clustering_teste.pkl")
# with open(file_path, 'rb') as file:
#     clustering : ImageClustering = pickle.load(file)

#     for c in clustering.clusters[:5]:
#         #print(Image(c.image_path, clustering.h).path)
#         print(c.image_path)
#         print(c.color_reference, '\n')

    

#     densities = [cluster.density_reference for cluster in clustering.clusters]
#     colors = [cluster.color_reference for cluster in clustering.clusters]

#     n = 5
#     images = [Image(img, clustering.h) for cluster in clustering.clusters
#               for img in cluster.members[:n]]
    
#     densities = [img.black_pixel_density_value for img in images]
#     colors = [img.most_common_color for img in images]
#     del images
    
#     sorted_data = sorted(zip(colors, densities), key=lambda x: x[0])

#     colors, densities = zip(*sorted_data)
#     print(len(colors), len(densities))

#     color_to_index = {}
#     index = 0
    
#     numerical_colors = []
#     for color in colors:
#         if color not in color_to_index:
#             color_to_index[color] = index
#             index += 1
#         numerical_colors.append(color_to_index[color])

#     normalized_colors = np.array(colors) / 255.0
#     cmap = ListedColormap(normalized_colors)
    
#     plt.figure(figsize=(16, 24))
#     plt.scatter(range(len(densities)), densities, c=numerical_colors, cmap=cmap, s=100, alpha=0.75)
#     plt.xlabel('Index')
#     plt.ylabel('Black Pixel Density')
#     plt.title('Numerical Colors and Floats')
#     plt.colorbar(label='Numerical Color', orientation="horizontal")
#     plt.show()


# img1 = Image(all_images[0], 10)
# img2 = Image(all_images[3], 10)

# cv2.imshow("Cropped Image", img)
# cv2.waitKey(0)


# img =  cv2.imread(file_path)
# print(img.shape)
# _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# cv2.imshow('Imagem binária', binary_img)
# cv2.waitKey(0)
#print(find_first_black_pixel_height(img))