from model.models import split, predict
from conf.conf import logging, settings
from connector.data import get_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import argparse


parser = argparse.ArgumentParser("Getting model params")

parser.add_argument('--values', 
                    nargs='+', 
                    type=float, 
                    required=True, 
                    help='Provide input values')

parser.add_argument('--model_path', 
                    type=str, 
                    help='Provide path to .pkl file with model')

args = parser.parse_args()

values = args.values
default_model_path = settings.MODEL.ranf_conf
model_path = args.model_path if args.model_path else default_model_path

print(predict(values, model_path))