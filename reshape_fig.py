from PIL import Image
import glob
import os
from pathlib import Path
import cv2
# path = "C:\\Users\\hawke\\OneDrive\\デスクトップ\\python\\汚れ検知\\cycle-gan-master\\data\\raindrop\\trainA\\"
# img_size = 256

path = "C:\\Users\\hawke\\OneDrive\\デスクトップ\\python\\cerema\\ESRGAN-master\\LR\\"
img_size = 120

for f in Path(path).rglob('*'):
   f.rename(path + f.stem + '.jpg')

filename_list = []
for f in glob.glob(path+'\\*'):
    filename_list.append(os.path.split(f)[1])

# print(filename_list)
for i in range(len(filename_list)):
    im = Image.open(path + filename_list[i])
    img_resize = im.resize((img_size, img_size))
    print(path)
    img_resize.save(path + 'Re_' + filename_list[i])

