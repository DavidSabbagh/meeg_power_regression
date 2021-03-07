import pandas as pd
import numpy as np

X = np.load("all_scores_mag_compo60.npy", allow_pickle=True).item()
df = pd.DataFrame(X)
df.to_csv("scores_mag_compo60.csv")
