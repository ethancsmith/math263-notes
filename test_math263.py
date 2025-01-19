import math263
import numpy
import yaml
import unittest

# test input IVP parameters
f = lambda x, y: x**2 - y;
a, b = 0, 2;
y0 = 3;
n = 10;

# read expected outputs from file
with open("test_data.yaml", "r") as fp:
    test_dict = yaml.safe_load(fp);

class TestSolvers(unittest.TestCase):

    def test_euler(self):
        (x_obs, y_obs) = math263.euler(f, a, b, y0, n);
        x_exp = test_dict["euler test"]["xi"];
        y_exp = test_dict["euler test"]["yi"];
        self.assertTrue(numpy.allclose(x_obs, x_exp), "math263.euler produces incorrect x-values on test")
        self.assertTrue(numpy.allclose(y_obs, y_exp), "math263.euler produces incorrect y-values on test")

    def test_mem(self):
        (x_obs, y_obs) = math263.mem(f, a, b, y0, n);
        x_exp = test_dict["mem test"]["xi"];
        y_exp = test_dict["mem test"]["yi"];
        self.assertTrue(numpy.allclose(x_obs, x_exp), "math263.mem produces incorrect x-values on test")
        self.assertTrue(numpy.allclose(y_obs, y_exp), "math263.mem produces incorrect y-values on test")

if __name__ == '__main__':
    unittest.main()