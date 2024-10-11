
from data_analyze import process_images
import matplotlib.pyplot as plt



    # Задайте пути к изображениям
file_path1 = "database/S2A_MSIL1C_20180919T083621_N0206_R064_T36UXA_20180919T105540.SAFE/GRANULE/L1C_T36UXA_A016935_20180919T084300/IMG_DATA/T36UXA_20180919T083621_B04.jp2"
file_path2 = "database/S2A_MSIL1C_20180919T083621_N0206_R064_T36UXA_20180919T105540.SAFE/GRANULE/L1C_T36UXA_A016935_20180919T084300/IMG_DATA/T36UXA_20180919T083621_B07.jp2"
 #
# Обработка изображений и нахождение ключевых точек
result_image = process_images(file_path1, file_path2)

# Визуализация результата
plt.figure(figsize=(10, 10))
plt.imshow(result_image, cmap='gray')

plt.axis('off')
plt.show()