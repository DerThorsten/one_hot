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


    def test_example2(self):
        dna = 'aacgtt'
        dna = numpy.fromstring(dna, dtype='uint8')


        mapping = numpy.zeros(256,dtype='uint8')
        mapping[ord('a')] = 0
        mapping[ord('c')] = 1
        mapping[ord('g')] = 2
        mapping[ord('t')] = 3

        res = one_hot.one_hot(dna,mapping, 4)
        print(res.shape)
        assert res[0,0] == 1
        assert res[0,1] == 0
        assert res[0,2] == 0
        assert res[0,3] == 0

        assert res[1,0] == 1
        assert res[1,1] == 0
        assert res[1,2] == 0
        assert res[1,3] == 0

        assert res[2,0] == 0
        assert res[2,1] == 1
        assert res[2,2] == 0
        assert res[2,3] == 0

        assert res[3,0] == 0
        assert res[3,1] == 0
        assert res[3,2] == 1
        assert res[3,3] == 0

        assert res[4,0] == 0
        assert res[4,1] == 0
        assert res[4,2] == 0
        assert res[4,3] == 1

        assert res[5,0] == 0
        assert res[5,1] == 0
        assert res[5,2] == 0
        assert res[5,3] == 1

        print(res)


