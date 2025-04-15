
import argparse
import pathlib
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input-train-data', type=pathlib.Path, required=True)
parser.add_argument('--input-test-data', type=pathlib.Path, required=True)
parser.add_argument('--input-hparams', type=pathlib.Path, required=True)
parser.add_argument('--output-model', type=pathlib.Path, required=True)
args = parser.parse_args()

train_data_path = args.input_train_data / 'data.npz'
test_data_path = args.input_test_data / 'data.npz'
hparam_path = args.input_hparams / 'hparams.yaml'

# load data
with open(train_data_path, 'rb') as fp:
    npz = np.load(fp)
    train_x = npz['x']
    train_y = npz['y']

with open(test_data_path, 'rb') as fp:
    npz = np.load(fp)
    test_x = npz['x']
    test_y = npz['y']

# load hparams for optimization
with open(hparam_path, 'r') as fp:
    opt_hparams = yaml.load(fp.read(), Loader=yaml.Loader)

# optimize hyperparameter
bestdeg = None
bestcoef = None
bestresid = None
for deg in range(opt_hparams['min_deg'], opt_hparams['max_deg']+1):
    # fitting polynomial to the data
    poly, (resid_train, _, _, _) = \
        np.polynomial.polynomial.Polynomial.fit(train_x, train_y, deg, full=True)
    poly = poly.convert()
    # calculate residual on test
    yhat = poly(test_x)
    resid_test = np.sum((test_y - yhat) ** 2)
    # compare to the best degree
    print(f'degree={deg} fit result: coef={poly.coef.tolist()}, '
          f'resid@train={resid_train}, resid@test={resid_test}')
    if bestdeg is None or resid_test < bestresid:
        bestdeg = deg
        bestcoef = poly.coef
        bestresid = resid_test

# write model and hyperparameters
if not args.output_model.exists():
    args.output_model.mkdir(parents=True)
np.savez(args.output_model / 'model.npz', coef=bestcoef)
with open(args.output_model / 'hparams.yaml', 'w') as fp:
    fp.write(f'deg: {bestdeg}')
