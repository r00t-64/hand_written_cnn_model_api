import keras
import tensorflow as tf

class CnnModel:
    def __init__(self):
        self.model = None
        self.model_architecture = 'model_architecture.json'
        self.model_weights = 'model_weights.h5'

    async def load_model(self):
        try:
            # Load the model architecture
            self.model = keras.models.model_from_json(open(self.model_architecture, 'r').read())
            # Load the model weights
            self.model.load_weights(self.model_weights)
        except Exception as error:
            print('Error loading the model:', error)
            raise Exception('Failed to load the model.')

    def predict(self, input_data):
        if not self.model:
            raise Exception('Model not loaded. Call load_model() first.')

        # Perform the forward pass to get predictions
        predictions = self.model.predict(tf.convert_to_tensor([input_data]))
        return predictions.tolist()
