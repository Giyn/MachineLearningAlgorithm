# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:08:04 2020

@author: Giyn
"""

from C45 import C45

if __name__ == "__main__":
    c1 = C45("iris.data", "iris.names")
    c1.fetchData()
    c1.preprocessData()
    c1.generateTree()
    c1.printTree()