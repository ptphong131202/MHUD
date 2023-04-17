import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

names = ["class", 'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged',
         'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo',
         'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild',
         'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay',
         'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots', 'seed', 'mold-growth',
         'seed-discolor', 'seed-size', 'shriveling', 'roots']
df = pd.read_csv("soybean-large.csv", names=names)

print("\n\tTAP DU LIEU TRUOC KHI CHUYEN DOI\n")
print(df)


df = df.drop(['date'], axis=1)
# Thay thế giá trị '?' bằng 0
df.replace('?', 0, inplace=True)


df.to_csv("soybean-update.csv")
