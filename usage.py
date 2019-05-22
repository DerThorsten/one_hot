import one_hot
import numpy


labels = numpy.array([0,1,1,2,2,3])
res = one_hot.one_hot(labels, 4)
print(res)