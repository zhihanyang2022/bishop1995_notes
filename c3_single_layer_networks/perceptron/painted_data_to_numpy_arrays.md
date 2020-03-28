# Painted data to Numpy arrays

This is a tutorial on how to convert data painted in Orange (saved as csv) into numpy arrays.

The toy data set here is `data_not_linearly_separable.csv`.

Read in data and do a sanity check:

```python
df = pd.read_csv('data_not_linearly_separable.csv')
df.head()
```

Numerical values are stored as strings in a csv. 

First, convert strings that represent numerical values into numerical values and then into a numpy array:

```python
samps = df[['x', 'y']][2:].convert_objects(convert_numeric=True).to_numpy()
```

Then, convert labels (strings) into a numpy array and then into numerical values:

```python
labels = df['Class'][2:].to_numpy()
labels[np.where(labels == 'C1')[0]] = -1
labels[np.where(labels == 'C2')[0]] = 1
```

