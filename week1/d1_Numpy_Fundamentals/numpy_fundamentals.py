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


