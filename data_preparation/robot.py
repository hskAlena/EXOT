import pickle
import pandas as pd

data = 3
with open('robot-cam.pkl', 'wb') as f:
	pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
