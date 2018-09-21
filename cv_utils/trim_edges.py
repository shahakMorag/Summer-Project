from os import listdir, path
from tqdm import tqdm
import cv2

dir_path = 'C:\Tomato_Classification_Project\Tomato_Classification_Project\cropped_data\size_500_stride_500'
save_dir = 'C:\Tomato_Classification_Project\Tomato_Classification_Project\cropped_data\cropped'
for file in tqdm(listdir(dir_path)):
    file_name = path.join(dir_path, file)
    image = cv2.imread(file_name,1)
    height, width = image.shape[:2]
    cropped = image[64:height-64][64:width-64]
    cv2.imwrite(path.join(save_dir,file), cropped)