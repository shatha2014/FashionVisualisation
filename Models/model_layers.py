from keras.models import load_model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, metavar='Path',
                    help='path of model')
Path = parser.parse_args()

def get_model_layers(path):
    model = load_model(path)
    for layer in model.layers:
        try:
            print(layer.name, layer.get_weights()[0].shape)
        except:
            print(layer.name)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, metavar='Path',
                    help='path of model')
    Path = parser.parse_args().model_path
    get_model_layers(Path)
