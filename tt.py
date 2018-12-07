from data import ImageFolder
from PIL import Image

# image_names = ImageFolder('inputs/', transform=None, return_paths=True)
# for i in image_names:
# 	print(i)
img = Image.open('inputs/008694.jpg').convert('RGB')
print(img)
