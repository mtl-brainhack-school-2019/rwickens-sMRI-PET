import pickle
import numpy
import pandas as pd
with open("data.pkl", "rb") as f:   
    loaded_pickle = pickle.load(f)
print(loaded_pickle)    
panda = pd.DataFrame(load_pickle, sep='\t') 
#print(magic)