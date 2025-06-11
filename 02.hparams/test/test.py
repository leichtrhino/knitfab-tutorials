
import argparse
import pathlib
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=pathlib.Path, required=True)
parser.add_argument('--input-model', type=pathlib.Path, required=True)
args = parser.parse_args()

data_path = args.input_data / 'data.npz'
model_path = args.input_model / 'model.npz'
hparam_path = args.input_model / 'hparams.yaml'

# load data
with open(data_path, 'rb') as fp:
    npz = np.load(fp)
    x = npz['x']
    y = npz['y']

# load hparams
with open(hparam_path, 'r') as fp:
    hparams = yaml.load(fp.read(), Loader=yaml.Loader)

# load model
with open(model_path, 'rb') as fp:
    npz = np.load(fp)
    model_coef = npz['coef']

# create initial model using hyperparameter
coef = [0.0] * (hparams['deg'] + 1)

# set model parameter (coef) and create polynomial from it
for i in range(hparams['deg']+1):
    coef[i] = model_coef[i]
poly = np.polynomial.Polynomial(coef)

# infer and calculate residuals
yhat = poly(x)
resid = np.sum((y - yhat) ** 2)
print(f'test result: resid={resid}')

