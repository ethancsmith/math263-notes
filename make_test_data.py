import numpy as np
import yaml

import math263

# scalar test input parameters
f = lambda x, y: x**2 - y
a, b = 0, 2
y0 = 3
n = 10

test_dict = dict()

# make data
(xi, yi) = math263.euler(f, a, b, y0, n)
test_dict["euler scalar test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.mem(f, a, b, y0, n)
test_dict["mem scalar test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.bem(f, a, b, y0, n)
test_dict["bem scalar test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.rk4(f, a, b, y0, n)
test_dict["rk4 scalar test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.ab2(f, a, b, y0, n)
test_dict["ab2 scalar test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.abm2(f, a, b, y0, n)
test_dict["abm2 scalar test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

# vector test input parameters
f = lambda t, r: np.array([r[0] + r[2], r[0] + r[1], -2 * r[0] - r[2]])
a, b = 0, 2 * np.pi
r0 = np.array([1, -1 / 2, -1])
n = 10

# make data
(xi, yi) = math263.euler(f, a, b, r0, n)
test_dict["euler vector test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.mem(f, a, b, r0, n)
test_dict["mem vector test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.bem(f, a, b, r0, n)
test_dict["bem vector test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.rk4(f, a, b, r0, n)
test_dict["rk4 vector test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.ab2(f, a, b, r0, n)
test_dict["ab2 vector test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

(xi, yi) = math263.abm2(f, a, b, r0, n)
test_dict["abm2 vector test"] = {"xi": xi.tolist(), "yi": yi.tolist()}

# write date to YAML file
with open("test_data.yaml", "w") as fp:
    yaml.safe_dump(test_dict, fp)
