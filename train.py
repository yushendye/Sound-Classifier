import tensorflow as tf
from model import *
from datasets import *
import pandas as pd
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_dir', type=str, default = None, help = 'Directory where you want to save the weights')
  parser.add_argument('--epochs', type=int, default = 100, help = 'number of epochs, default = 100')
  parser.add_argument('--dataset_dir', type=str, default=None, help = 'Location of the dataset folder')
  opt = parser.parse_args()

  model = get_model()
  X_train, X_test, y_train, y_test = get_dataset(opt.dataset_dir)

  train_model(X_train, y_train, X_test, y_test, opt.epochs, opt.checkpoint_dir)
