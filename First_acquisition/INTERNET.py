from keras.models import load_model
import joblib
import numpy as np
import tensorflow as tf
from scipy import stats


part = "C:/Users/atSine/OneDrive/Project/BCI/BCI-code/OpenTopic_BCI/First_acquisition/"

class INTERNET:
    def __init__(self, sample_rate,control_freq ,folder_path) -> None:
        print(folder_path)
        
        self.scaler = joblib.load(part+'models/' + folder_path + '/standard_scaler.pkl')
        print("Scaler loaded successfully.")

        self.model = load_model(part+'models/' + folder_path + '/' + folder_path + '.h5')

        print("Model loaded successfully.")

        self.output_freq = control_freq

        self.sample_rate = sample_rate
        self.buffer = np.array([])
        self.batched_set = np.array([])
        self.control_period = int(sample_rate/control_freq)
        self.batched_set_count = 0

        self.mode = -99
    
    def predict(self, X):
        
        if self.buffer.size == 0:
            self.buffer = np.array([X])

        elif len(self.buffer) < self.sample_rate:
            self.buffer = np.vstack([self.buffer, X])

        else:
            # If the buffer is full, implement a circular buffer
            self.buffer = np.roll(self.buffer, -2)  # Shift elements to the left
            self.buffer[-1] = X  # Add new data to the end


            if self.batched_set.size == 0:
                self.batched_set = np.array([self.scaler.transform(self.buffer.copy())])

            elif len(self.batched_set) < self.control_period:
                self.batched_set = np.vstack([self.batched_set, [self.scaler.transform(self.buffer.copy())]])
                
            else:
                self.batched_set_count += 1

                self.batched_set = np.roll(self.batched_set, -self.sample_rate*2)  # Shift elements to the left
                self.batched_set[-1] = self.scaler.transform(self.buffer.copy())  # Add new data to the end
                
                
                if(self.batched_set_count > self.sample_rate):
                         # print(batched_set)
                    pred = self.model.predict(self.batched_set ,batch_size=self.control_period,verbose=0)
                    # print(pred)

                    one_hot = np.argmax(pred, axis=1)

                    self.mode = stats.mode(one_hot)[0]
                    self.batched_set_count = 0
                
        return self.mode




import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")