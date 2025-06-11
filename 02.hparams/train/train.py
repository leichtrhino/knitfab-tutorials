
import argparse
import pathlib
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=pathlib.Path, required=True)
parser.add_argument('--input-hparams', type=pathlib.Path, required=True)
parser.add_argument('--output-model', type=pathlib.Path, required=True)
args = parser.parse_args()

data_path = args.input_data / 'data.npz'
hparam_path = args.input_hparams / 'hparams.yaml'

# load data
with open(data_path, 'rb') as fp:
    npz = np.load(fp)
    x = npz['x']
    y = npz['y']

# load hparams
with open(hparam_path, 'r') as fp:
    hparams = yaml.load(fp.read(), Loader=yaml.Loader)

# fitting polynomial to the data
poly, (resid, _, _, _) = \
    np.polynomial.polynomial.Polynomial.fit(x, y, deg=hparams['deg'], full=True)
poly = poly.convert()
print(f'fit result: coef={poly.coef}, resid={resid}')

# write model and hyperparameters
if not args.output_model.exists():
    args.output_model.mkdir(parents=True)
np.savez(args.output_model / 'model.npz', coef=poly.coef)
with open(hparam_path, 'rb') as fp:
    hparams_raw = fp.read()
with open(args.output_model / 'hparams.yaml', 'wb') as fp:
    fp.write(hparams_raw)
