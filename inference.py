
import cv2
import matplotlib.pyplot as plt
from image_analysis import process_images

def main():
    file_path1 = "S2A_MSIL1C_20180919T083621_N0206_R064_T36UXA_20180919T105540.SAFE/GRANULE/L1C_T36UXA_A016935_20180919T084300/IMG_DATA/T36UXA_20180919T083621_B04.jp2"
    file_path2 = "S2B_MSIL1C_20180805T083559_N0206_R064_T36UXA_20180805T123757.SAFE/GRANULE/L1C_T36UXA_A007383_20180805T084554/IMG_DATA/T36UXA_20180805T083559_B11.jp2"
    result_image  = process_images(file_path1, file_path2)


    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()