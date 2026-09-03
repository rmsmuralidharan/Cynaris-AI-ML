import numpy as np

### checking the version of the numpy library

print(f"numpy_version: {np.__version__}" )


## 1D array - multiple elements stored in one dimension (one axis)
array_1d = np.array([10,20,30,40,50])

##printing the output of our array
print('1D Array:')
print(array_1d)

## executing the shape for the 1d array
print('shape:', array_1d.shape)

### 2d array - multiple collection of elements stored in 2 dimensions (2 axis)
array_2d = np.array([
    [1,2,3],
    [4,5,6]
]) 

## printing its output
print("2D array:")
print(array_2d)

## executing the shape of the 2d array
print('shape:', array_2d.shape)


### 3d array - multiple collection of elements stored in 3 dimensions (3 axis)
array_3d = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

## printing its output
print('3d array:')
print(array_3d)

## executing its shape of the 3d array

print('shape:', array_3d.shape)



## broadcasting - can perform operations on arrays of different shapes by expanding the smaller array when possible

matrix = np.array([
    [1,2,3],
    [4,5,6]
])

vector = np.array([10,20,30])

result = matrix + vector

print('\nMatrix:')
print(matrix)
print(matrix.shape)

print('\nVector:')
print(vector)
print(vector.shape)

print('\nBroadcasting')
print(result)
print(result.shape)


###vectorized operations - performs operations for entire elements instead of using loop

numbers = np.array([1,2,3,4,5])

result = numbers + 5

print('\nOriginal array:')
print(numbers)
print(numbers.shape)

print('\nAfter vectorizing by adding the elements each by 5:')
print(result)
print(result.shape)


## matrix multiplication

matrix_a = np.array([
    [1,2,3],
    [4,5,6]
])

matrix_b = np.array([
    [1,2],
    [4,5],
    [7,8]
])

result = np.matmul(matrix_a, matrix_b)

print('\nMatrix A:')
print(matrix_a)
print(matrix_a.shape)

print('\nMatrix B:')
print(matrix_b)
print(matrix_b.shape)

print('\nMatrix Multiplication:')
print(result)
print(result.shape)


### calculating mean, std, corelation on a students csv dataset

data = np.genfromtxt(   ### numpy function used to read numerical values from csv file
    'students.csv',
    delimiter=',',
    skip_header=1
)

print('\nDataset:')
print(data.shape)

### mean - average of both the columns

mean = np.mean(data, axis=0)

print('\nMean')
print(mean)

### standard deviation - it tells us how much values are spreadout from the mean

std = np.std(data, axis=0)

print('\nStandard Deviation')
print(std)

## corelation - it tells us how two features are related to each other

corr = np.corrcoef(data, rowvar=False) ## rowvar - parameter used for telling the python each column is a variable

print('\nCorelation Matrix:')
print(corr)
