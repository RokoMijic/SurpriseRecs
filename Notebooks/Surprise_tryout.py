# %% [markdown]
# **Ttrying out the Surpr!se Library**

# %%
import numpy as np
import matplotlib.pyplot as plt

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate


# %%
# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# %%
data.raw_ratings

# %%
algo = SVD()

# %%
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# %%
