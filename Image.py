import cv2
from collections import Counter

class Image:


    def __init__(self, image_path: str, height: float):
        self.path = image_path
        self.image_matrix = self.crop_image(height, None)
        self.black_pixel_density_value = self.black_pixel_density()
        self.most_common_color = self.color_mode()
    
    def crop_image(self, desired_height, desired_width):

        image = cv2.imread(self.path)
        height, width, _ = image.shape
        
        if not desired_height:
            desired_height = height
            
        if not desired_width:
            desired_width = width
            
        if (height >= desired_height) and (width >= desired_width):
            cropped_image = image[0:desired_height, 0:desired_width]
            return cropped_image
        else:
            return image
        
    def color_mode(self):
        image_rgb = cv2.cvtColor(self.image_matrix, cv2.COLOR_BGR2RGB)

        pixels = image_rgb.reshape(-1, 3)
        color_counts = Counter(map(tuple, pixels))

        most_common_color = color_counts.most_common(1)[0][0]

        return most_common_color

    def black_pixel_density(self):
        grayscale_image = cv2.cvtColor(self.image_matrix, cv2.COLOR_BGR2GRAY)
        
        total_pixels = grayscale_image.shape[0] * grayscale_image.shape[1]
        black_pixels = total_pixels - cv2.countNonZero(grayscale_image)
        
        black_pixel_density = black_pixels / total_pixels

        return black_pixel_density
