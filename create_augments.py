from PIL import Image
import os
import torchvision.transforms as transforms
import re

folder_path_original = "all_images_original" #has 300 images of 3 types: A, B, C
folder_path_augment = "all_images_augment" #starts empty

transform = transforms.Compose([
    # Geometric
    transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
    transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
    transforms.RandomApply([transforms.RandomResizedCrop(size=(240, 320), scale=(0.8, 1.0), ratio=(4/3, 4/3))], p=0.3),

    # Noise / blur / brightness / exposure
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),

    # Color-space
    transforms.RandomApply([transforms.ColorJitter(hue=0.1, saturation=0.2)], p=0.3),
])

img_idx = 0  # makes filenames unique across all images

#for each image in folder
for img_path in os.listdir(folder_path_original):
    #example file name path C:\Users\Anthony\Documents\CS490\code\all_images_original\pill~bottle-A_20260120_155305.jpg
    #grab class type from path
    
    match = re.search(r"-([A-C])_", img_path)
    bottle_class = match.group(1)
        
    #open the original image
    image_path = os.path.join(folder_path_original, img_path)
    image = Image.open(image_path).convert("RGB")
    out_path = os.path.join(folder_path_augment, f"{bottle_class}_{img_idx}_0.jpg")
    image.save(out_path)


    #make 2 augments
    for i in range(1,3):
        # Load image and apply transform if any
        augment_image = image.copy() # deepcopy(image)
        augment_image = transform(augment_image)
        
        #add augmented image to folder
        filename = f"{bottle_class}_{img_idx}_{i}.jpg"
        out_path = os.path.join(folder_path_augment, filename)
        
        #Save image 
        augment_image.save(out_path)

    img_idx += 1
        
        


