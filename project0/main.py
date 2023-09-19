import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    return np.random.rand(n,1)

#print(randomization(5))

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    A = np.random.rand(h,w)
    B = np.random.rand(h,w)
    s = A + B
    return A, B, s

#print(operations(2,3))

def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    return np.linalg.norm(A+B)

#print(norm(np.array([1,2]), np.array([3,4])))


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    return np.tanh(np.matmul(weights.T,inputs))

#print(neural_network(np.array([1,2]), np.array([3,4])))

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x <= y:
        return x*y
    else:
        return x/y
    
#print(scalar_function(2,1))

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    return np.vectorize(scalar_function)(x,y)

f=vector_function(np.array([1,2]), np.array([3,4]))
print(f)
g=vector_function(2, 1)
print(g)
print(vector_function(2,1))
