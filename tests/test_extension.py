import one_hot 
from unittest import TestCase
import numpy


class ExampleTest(TestCase):

    def test_example1(self):
        labels = numpy.array([0,0,1,2])
        res = one_hot.one_hot(labels, 3)
        print(res.shape)
        assert res[0,0] == 1
        assert res[0,1] == 0
        assert res[0,2] == 0

        assert res[1,0] == 1
        assert res[1,1] == 0
        assert res[1,2] == 0

        assert res[2,0] == 0
        assert res[2,1] == 1
        assert res[2,2] == 0

        assert res[3,0] == 0
        assert res[3,1] == 0
        assert res[3,2] == 1
        print(res)


