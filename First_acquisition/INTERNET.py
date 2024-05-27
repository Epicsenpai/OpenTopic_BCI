from keras.models import load_model
import joblib
import numpy as np
import tensorflow as tf

class INTERNET:
    def __init__(self, folder_path) -> None:
        print(folder_path)
        
        self.scaler = joblib.load('models/' + folder_path + '/standard_scaler.pkl')
        print("Scaler loaded successfully.")

        self.model = load_model('models/' + folder_path + '/' + folder_path + '.h5')

        print("Model loaded successfully.")
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)


        expanded_np_array = np.expand_dims(X_scaled, axis=0)  # Now shape is (1, 250, 2)

        # Step 3: Convert the expanded NumPy array to a TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(expanded_np_array, dtype=tf.float64)
        print(tf_tensor.shape)
        pred = self.model.predict(tf_tensor)
        print(pred)
        return np.argmax(pred, axis=1)
