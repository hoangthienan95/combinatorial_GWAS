import argparse

parser = argparse.ArgumentParser(description='Tune and train a model')

parser.add_argument('--phenotypes', '-p', nargs='+', required=True)
parser.add_argument('--networks', '-n', nargs='+', choices=['CNN', 'DNN', 'cnn', 'dnn', 'deeper-dnn', 'deeper-cnn'], required=True)
parser.add_argument('--dates', '-d', nargs='+', required=True)
# Unfortunately to avoid intricacies of parsing floating point numbers I had to compromise here a little
# Pass in INTEGERS denoting the negative power of ten of the threshold i.e. threshold_1e-08 == 8, threshold_1e-6 == 6
parser.add_argument('--thresholds', '-t', type=int, nargs='+', required=True)
parser.add_argument('--max-samples', '-m', type=int, nargs='+', required=True)
parser.add_argument('--output-file', default=None)

args = parser.parse_args()

globals().update(vars(args))

networks = [network.lower() for network in networks]

import tensorflow as tf
import shap
import os
import numpy as np
import pickle
import tqdm
import multiprocessing as mp
from combinatorial_gwas.genotype import load_genetic_file

def get_output_datapath(model, nt, dt, p, t, s):
    out = f"/lab/corradin_biobank/FOR_AN/combinatorial_GWAS/data/{model}/{nt}/{dt}/{output_file or f'phenotype_{p}_threshold_1e-0{t}_max-samples_{s}'}"
    print(model, f"'{out}'")
    return out

genome_files = {chromosome: load_genetic_file(chromosome) for chromosome in tqdm.tqdm(list(range(1, 23)), "Loading genotype file(s)")}

def load_datasource(nt, dt, p, t, s):
    with open(get_output_datapath('07_model_output', nt, dt, p, t, s), 'rb') as f:
        out = pickle.load(f)['datasource']
        out.genome_files = genome_files
        return out

def shap_scores(model,x_train,x_test):
    #Assert correct input shape of x_train
    explainer = shap.DeepExplainer(model,x_train)
    shap_values = explainer.shap_values(x_test)
    return shap_values

def get_data(ds):
    x, y = ds.get_data(slice(0, None), 'test')
    indices = np.unique(np.nonzero(y))
    x, y = x[indices, :, :], y[indices]
    return (*ds.get_data(slice(0, y.shape[0]), 'train_tune'), x, y)

def load_model(nt, dt, p, t, s):
    return tf.keras.models.load_model(get_output_datapath('06_models', nt, dt, p, t, s))

def get_major_indices(shap_scores):
    ys = [np.unique(np.nonzero(score)[1]) for score in shap_scores]
    nonzeros = [score[:, y, :] for y, score in zip(ys, shap_scores)]
    l2s = [y[np.argsort(np.sum(np.sum(nonzero ** 2, axis=2), axis=0))] for y, nonzero in zip(ys, nonzeros)]
    l1s = [y[np.argsort(np.sum(np.sum(np.abs(nonzero), axis=2), axis=0))] for y, nonzero in zip(ys, nonzeros)]
    return ys, nonzeros, l2s, l1s

def _driver(nt, dt, p, t, s):
    model = load_model(nt, dt, p, t, s)
    ds = load_datasource(nt, dt, p, t, s)
    rt_x, rt_y, tx, ty = get_data(ds)
    shap_values = shap_scores(model, rt_x, tx)
    ys, nonzeros, l2s, l1s = get_major_indices(shap_values)
    return {"shap_values":shap_values, "ys":ys, "nonzeros":nonzeros, "l2s":l2s, "l1s":l1s}

def driver(driver_args):
    try:
        nt, dt, p, t, s = driver_args
        to_save = {"args":vars(args), "interpreted":_driver(nt, dt, p, t, s)}
        out = get_output_datapath("08_reporting", nt, dt, p, t, s)
        os.makedirs(out[:out.rindex("/")], exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(to_save, f)
    except FileNotFoundError:
        print(f"Error: Could not find model or data file for args {driver_args}")



for driver_args in ((nt, dt, p, t, s) for nt in networks for dt in dates for p in phenotypes for t in thresholds for s in max_samples):
    driver(driver_args)