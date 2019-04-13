from PIL import Image
import os
import argparse

def resize(path):
    for item in os.listdir(path):
        item_path = os.path.join(path + item)

        if os.path.isfile(item_path):
            try:
                im = Image.open(item_path)
                imResize = im.resize((224, 224))
                imResize.save(item_path, 'JPEG')
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, metavar='Path',
                        help='path of folder with images')
    Path = parser.parse_args().folder_path
    print(Path)
    resize(Path)