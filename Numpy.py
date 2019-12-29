import numpy as np
##l = [[1,2,3],[4,5,6]]
##
##npArray = np.array(l)
##print(npArray)
##print(npArray.shape)
##print(npArray.dtype)
##
##print(np.zeros((2,2),dtype =np.int16))
##
##print(np.ones((2,3),dtype = np.int16))
##
##l1 = np.array([(34,54,6),(32,43,11)])
##print(l1)
##
##k = l1.reshape(3,2)
##print(k)
##
##k = l1.flatten()
##print(k)
##
##print(np.hstack((npArray,l1)))
##print(np.vstack((npArray,l1)))
##
##print(np.random.normal(5,0.5,10))
##
##A = np.matrix(np.ones((4,4),dtype = np.int16))
##print(A)
##
##np.asarray(A)[2] = 2
##print(A)
##
##print(np.arange(1,11,3))
##print(np.linspace(1,5,num=10,endpoint = False))
##print(np.logspace(3,4,num=10,endpoint = False))
##print(l1.itemsize)


##Indexing and Slicing ##


a = np.array([(1,2,3),(4,5,6)])
print(a[0])

print(a[0,2])

##print(a[:,1]) : means all rows of that particular column 

##print(a[0, :2]) To return the first two values of the second row. You use : to select all columns up to the second

##############################



f = np.array([[1,2],[3,5]])
g = np.array([[4,5],[7,8]])

print(np.dot(f,g))

print(np.matmul(f,g))

##print(np.linalg.det(f))  Determinant

print(np.size(f))











