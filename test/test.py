#!/usr/bin/python

import libfann

ann = libfann.neural_net()
ann.create_from_file("xor.net")

print ann.run([1, -1])
