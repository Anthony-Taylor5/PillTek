import re

path = r"C:\Users\Anthony\Documents\CS490\code\all_images_original\pill~bottle-A_20260120_155305.jpg"

match = re.search(r"-([A-C])_", path)

if match:
    letter = match.group(1)
    print(letter)

