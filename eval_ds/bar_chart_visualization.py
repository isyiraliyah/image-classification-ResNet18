import os
import matplotlib.pyplot as plt

categories = ['safe', 'violent']
counts = [len(os.listdir('eval_ds/safe')), len(os.listdir('eval_ds/violent'))]

plt.bar(categories, counts)
plt.title("Dataset Distribution")
plt.ylabel("Number of Images")
plt.show()
