This project is the code of the paper "Satellite-ground synchronous in-situ dataset of water parameters for typical lakes in China".
quality control.py is the quality control code for five parameters (remote sensing reflectance (Rrs) and water parameter data (chlorophyll-a (Chl-a), total suspended matter (TSM), Secchi disk depth (SDD))).
retrieval_model.py is the code for a random forest regression model in which Cha-a, TSM and SDD parameters are simulated and retrieved.

pyhton version: 3.8.18

Required python libraries:
import numpy
import pandas
import sklearn
import json
import pickle
import matplotlib