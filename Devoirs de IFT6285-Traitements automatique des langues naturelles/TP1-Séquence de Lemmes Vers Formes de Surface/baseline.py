"""
This script computes the baseline results. The output of these results can be found in the report.
"""

import loader
import metric


FILES_LARGE_CORPUS = 300

x, y = loader.load("data/test")
accuracy = metric.accuracy(y, x)
metric.display_result(len(x), accuracy)

x, y = loader.load("data/train", FILES_LARGE_CORPUS)
accuracy = metric.accuracy(y, x)
metric.display_result(len(x), accuracy)
