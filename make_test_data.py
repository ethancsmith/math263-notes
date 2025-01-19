import math263
import yaml

# test input parameters
f = lambda x, y: x**2 - y;
a, b = 0, 2;
y0 = 3;
n = 10;

test_dict = dict();

# make data
(xi, yi) = math263.euler(f, a, b, y0, n);
test_dict["euler test"] = {"xi": xi.tolist(), "yi": yi.tolist()};

(xi, yi) = math263.mem(f, a, b, y0, n);
test_dict["mem test"] = {"xi": xi.tolist(), "yi": yi.tolist()};

(xi, yi) = math263.bem(f, a, b, y0, n);
test_dict["bem test"] = {"xi": xi.tolist(), "yi": yi.tolist()};

(xi, yi) = math263.rk4(f, a, b, y0, n);
test_dict["rk4 test"] = {"xi": xi.tolist(), "yi": yi.tolist()};

(xi, yi) = math263.ab2(f, a, b, y0, n);
test_dict["ab2 test"] = {"xi": xi.tolist(), "yi": yi.tolist()};

(xi, yi) = math263.abm2(f, a, b, y0, n);
test_dict["abm2 test"] = {"xi": xi.tolist(), "yi": yi.tolist()};

with open("test_data.yaml", "w") as fp:
    yaml.safe_dump(test_dict, fp);