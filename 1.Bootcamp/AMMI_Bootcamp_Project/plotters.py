import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

headers = ['ID', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'class']

df = pd.read_csv('Data/train_data.txt', header=None)
df.columns = headers



def scatter_selves(data, head):
  plt.figure(figsize=(9,6),dpi=150)
  ticks = [0,.25,0.5,0.75,1]
  for i in range(1,len(head[1:-1])+1):
    plt.subplot(3,4,i)
    plt.scatter(data.loc[data['class'] == 1][head[i]], data.loc[data['class'] == 1][head[i]], label=1, s=1)
    plt.scatter(data.loc[data['class'] == 2][head[i]], data.loc[data['class'] == 2][head[i]], label=2, s=1)
    if i < 9: plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if (i != 1) and (i != 5) and (i != 9): plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    plt.title(f'Feature {i}')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(linestyle='--')
    plt.legend()
  plt.suptitle(f'Features vs. Them selves')
  plt.show()

# scatter_selves(df, headers)


def scatter_others(data, head, n):
  plt.figure(figsize=(9,6),dpi=150)
  ticks = [0,.25,0.5,0.75,1]
  j = 1
  for i in range(1,len(head[1:-1])+1):
    # if i != n:
    plt.subplot(3,4,j)
    plt.scatter(data.loc[data['class'] == 1][head[n]], data.loc[data['class'] == 1][head[i]], label=1, s=1)
    plt.scatter(data.loc[data['class'] == 2][head[n]], data.loc[data['class'] == 2][head[i]], label=2, s=1)
    if i < 9: plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if (i != 1) and (i != 5) and (i != 9): plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    plt.title(f'Feature {i}')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(linestyle='--')
    plt.legend()
    j += 1
  plt.suptitle(f'Feature {n} vs. Other Features')
  plt.show()

for i in range(1,12):
    scatter_others(df, headers, i)


def scatter_nm(data, head, n, m):
  plt.figure(figsize=(9,6),dpi=150)
  ticks = [0,.25,0.5,0.75,1]
  j = 1
  for i in range(1,len(head[1:-1])+1):
    plt.subplot(3,4,j)
    plt.scatter(data.loc[data['class'] == 1][head[n]]*data.loc[data['class'] == 1][head[m]], data.loc[data['class'] == 1][head[i]], label=1, s=5, marker='^')
    plt.scatter(data.loc[data['class'] == 2][head[n]]*data.loc[data['class'] == 2][head[m]], data.loc[data['class'] == 2][head[i]], label=2, s=5, marker='s')
    if i < 9: plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if (i != 1) and (i != 5) and (i != 9): plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    plt.title(f'Feature {i}')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(linestyle='--')
    plt.legend()
    j += 1
  plt.suptitle(f'Features: {n}*{m} vs. Other Features')
  plt.show()
# for j in range(1, 12):
#     for i in range(j, 12):
#         scatter_nm(df, headers, j, i)
# scatter_nm(df, headers, j, i)

def scatter_nplusm(data, head, n, m):
  plt.figure(figsize=(9,6),dpi=150)
  ticks = [0,.25,0.5,0.75,1]
  j = 1
  for i in range(1,len(head[1:-1])+1):
    plt.subplot(3,4,j)
    plt.scatter(data.loc[data['class'] == 1][head[n]]+data.loc[data['class'] == 1][head[m]], data.loc[data['class'] == 1][head[i]], label=1, s=5, marker='^')
    plt.scatter(data.loc[data['class'] == 2][head[n]]+data.loc[data['class'] == 2][head[m]], data.loc[data['class'] == 2][head[i]], label=2, s=5, marker='s')
    if i < 9: plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if (i != 1) and (i != 5) and (i != 9): plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    plt.title(f'Feature {i}')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(linestyle='--')
    plt.legend()
    j += 1
  plt.suptitle(f'Features: {n}+{m} vs. Other Features')
  plt.show()
# for j in range(1, 12):
#     for i in range(j, 12):
#         scatter_nplusm(df, headers, j, i)
