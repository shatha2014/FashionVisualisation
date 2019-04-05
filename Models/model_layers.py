from keras.models import load_model


def get_model_layers(path):
    model = load_model(path)
    for layer in model.layers:
        try:
            print(layer.name, layer.get_weights()[0].shape)
        except:
            print(layer.name)
