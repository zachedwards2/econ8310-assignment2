import unittest
import statsmodels
import patsy as pt
import pandas as pd
import numpy as np
import sklearn
import statsmodels.discrete.discrete_model as dm

# Import your code from parent directory
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from assignment2 import model, modelFit, pred

# Run the checks

def tjurr(truth, pred):
    truth = list(truth)
    pred = list(pred)
    y1 = np.mean([y for x, y in enumerate(pred) if truth[x]==1])
    y2 = np.mean([y for x, y in enumerate(pred) if truth[x]==0])
    return y1-y2


class testCases(unittest.TestCase):
    def testAccuracy1(self):
        truth = pd.read_csv(currentdir + "/testData.csv")['meal']

        self.assertTrue(tjurr(truth, pred)>0.2,  "Your predictions have a Tjurr R-squared below 0.20.")
