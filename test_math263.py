import unittest

import numpy
import yaml

import math263

# read expected outputs from file
with open("test_data.yaml", "r") as fp:
    test_dict = yaml.safe_load(fp)

# test parameters for scalar IVP test
f = lambda x, y: x**2 - y


class ScalarTestODESolvers(unittest.TestCase):

    a, b = 0, 2
    y0 = 3
    n = 10

    def test_euler(self):
        (x_obs, y_obs) = math263.euler(f, self.a, self.b, self.y0, self.n)
        x_exp = test_dict["euler scalar test"]["xi"]
        y_exp = test_dict["euler scalar test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.euler produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.euler produces incorrect y-values on test",
        )

    def test_mem(self):
        (x_obs, y_obs) = math263.mem(f, self.a, self.b, self.y0, self.n)
        x_exp = test_dict["mem scalar test"]["xi"]
        y_exp = test_dict["mem scalar test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.mem produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.mem produces incorrect y-values on test",
        )

    def test_bem(self):
        (x_obs, y_obs) = math263.bem(f, self.a, self.b, self.y0, self.n)
        x_exp = test_dict["bem scalar test"]["xi"]
        y_exp = test_dict["bem scalar test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.bem produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.bem produces incorrect y-values on test",
        )

    def test_rk4(self):
        (x_obs, y_obs) = math263.rk4(f, self.a, self.b, self.y0, self.n)
        x_exp = test_dict["rk4 scalar test"]["xi"]
        y_exp = test_dict["rk4 scalar test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.rk4 produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.rk4 produces incorrect y-values on test",
        )

    def test_ab2(self):
        (x_obs, y_obs) = math263.ab2(f, self.a, self.b, self.y0, self.n)
        x_exp = test_dict["ab2 scalar test"]["xi"]
        y_exp = test_dict["ab2 scalar test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.ab2 produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.ab2 produces incorrect y-values on test",
        )

    def test_abm2(self):
        (x_obs, y_obs) = math263.abm2(f, self.a, self.b, self.y0, self.n)
        x_exp = test_dict["abm2 scalar test"]["xi"]
        y_exp = test_dict["abm2 scalar test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.abm2 produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.abm2 produces incorrect y-values on test",
        )


# test parameters for vector IVP test
vec_f = lambda t, r: numpy.array([r[0] + r[2], r[0] + r[1], -2 * r[0] - r[2]])


class VectorTestODESolvers(unittest.TestCase):

    a, b = 0, 2 * numpy.pi
    r0 = numpy.array([1, -1 / 2, -1])
    n = 10

    def test_euler(self):
        (x_obs, y_obs) = math263.euler(vec_f, self.a, self.b, self.r0, self.n)
        x_exp = test_dict["euler vector test"]["xi"]
        y_exp = test_dict["euler vector test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.euler produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.euler produces incorrect y-values on test",
        )

    def test_mem(self):
        (x_obs, y_obs) = math263.mem(vec_f, self.a, self.b, self.r0, self.n)
        x_exp = test_dict["mem vector test"]["xi"]
        y_exp = test_dict["mem vector test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.mem produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.mem produces incorrect y-values on test",
        )

    def test_bem(self):
        (x_obs, y_obs) = math263.bem(vec_f, self.a, self.b, self.r0, self.n)
        x_exp = test_dict["bem vector test"]["xi"]
        y_exp = test_dict["bem vector test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.bem produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.bem produces incorrect y-values on test",
        )

    def test_rk4(self):
        (x_obs, y_obs) = math263.rk4(vec_f, self.a, self.b, self.r0, self.n)
        x_exp = test_dict["rk4 vector test"]["xi"]
        y_exp = test_dict["rk4 vector test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.rk4 produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.rk4 produces incorrect y-values on test",
        )

    def test_ab2(self):
        (x_obs, y_obs) = math263.ab2(vec_f, self.a, self.b, self.r0, self.n)
        x_exp = test_dict["ab2 vector test"]["xi"]
        y_exp = test_dict["ab2 vector test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.ab2 produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.ab2 produces incorrect y-values on test",
        )

    def test_abm2(self):
        (x_obs, y_obs) = math263.abm2(vec_f, self.a, self.b, self.r0, self.n)
        x_exp = test_dict["abm2 vector test"]["xi"]
        y_exp = test_dict["abm2 vector test"]["yi"]
        self.assertTrue(
            numpy.allclose(x_obs, x_exp),
            "math263.abm2 produces incorrect x-values on test",
        )
        self.assertTrue(
            numpy.allclose(y_obs, y_exp),
            "math263.abm2 produces incorrect y-values on test",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
