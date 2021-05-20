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
parser.add_argument('--data-batch', '-db', type=int, default=50000)

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
from sklearn.linear_model import LogisticRegression as lgr
from combinatorial_gwas.simulation import SimulatedPheno, SNPInfoUnit
from combinatorial_gwas.genotype import load_genetic_file

def get_output_datapath(model, nt, dt, p, t, s):
    return f"/lab/corradin_biobank/FOR_AN/combinatorial_GWAS/data/{model}/{nt}/{dt}/phenotype_{p}_threshold_1e-0{t}_max-samples_{s}"

genome_files = {chromosome: load_genetic_file(chromosome) for chromosome in tqdm.tqdm(list(range(1, 23)), "Loading genotype file(s)")}
def load_datasource(nt, dt, p, t, s):
    with open(get_output_datapath('07_model_output', nt, dt, p, t, s), 'rb') as f:
        out = pickle.load(f)['datasource']
        out.genome_files = genome_files
        return out

def shap_scores(model,x_train,x_test):
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    explainer = shap.LinearExplainer(model,background)
    shap_values = explainer.shap_values(x_test)
    return shap_values

def get_data(ds):
    x, y = ds.get_data(slice(0, None), 'test')
    indices = np.unique(np.nonzero(y))
    x, y = x[indices, :, :], y[indices]
    return (*ds.get_data(slice(0, y.shape[0]), 'train_tune'), x, y)

def get_batched_train_test_data(datasource, phenotype, threshold):
    t_x, t_y = datasource.get_data(slice(0, None), 'test')
    r_x, r_y = batched_train_data(datasource, phenotype, threshold, 'train')
    rt_x, rt_y = datasource.get_data(slice(0, None), 'train_tune')
    return r_x, r_y, t_x, t_y

def acquire_data(args):
    phenotype, threshold = args
    datasource = get_datasource(phenotype, threshold)
    data = get_batched_train_test_data(datasource, phenotype, threshold)
    return data[:2], data[2:], datasource

def batched_train_data(datasource, phenotype, threshold, dataset):
    class TrainingData(tf.keras.utils.Sequence):
        def __init__(self, datasource, phenotype, threshold, dataset):
            self.datasource = datasource
            self.phenotype = phenotype
            self.threshold = threshold
            self.dataset = dataset
            self.batch_indeces = list(range(0, datasource.data_dict.get(dataset).shape[0], data_batch))
            if self.batch_indeces[-1] != datasource.max_samples:
                self.batch_indeces.append(datasource.max_samples)
            self.saved = [None] * len(self)
        
        def __len__(self):
            return len(self.batch_indeces) - 1
        
        def __getitem__(self, idx):
            import pickle
            os.makedirs(f'{test_directory}/train_cache', exist_ok=True)
            if self.saved[idx]:
                with open(self.saved[idx], 'rb') as f:
                    return pickle.load(f)
            else:
                data = self.datasource.get_data(slice(*self.batch_indeces[idx:idx+2]), self.dataset)
                self.saved[idx] = f'{test_directory}/train_cache/{f"phenotype_{self.phenotype}_threshold_{self.threshold}_max_samples_{max_samples}_dataset_{self.dataset}_{idx}.cache"}'
                with open(self.saved[idx], 'wb') as f:
                    pickle.dump(data, f)
            return data
        
        def __delitem__(self, idx):
            if self.saved[idx]:
                os.remove(self.saved[idx])
                self.saved[idx] = None
    return TrainingData(datasource, phenotype, threshold, dataset), None


def SNP_loop(x_train,x_test,y_train,num_samples):
    x_train = tf.constant()
    x_train = x_train.numpy()
    y_train = y_train.numpy()
    a,b,c = x_train.shape
    output_shap = np.zeros((num_samples,0,c))
    
    model_outputs = []
    
    for i in range(b):
        x_train1 = np.squeeze(x_train[:num_samples,i,:])
        x_test1 = np.squeeze(x_test[:num_samples,i,:])
        y_train1 = y_train[:num_samples,...]
        model = lgr().fit(x_train1,y_train1)
        scores = shap_scores(model,x_train1[:num_samples,...],x_test1[:num_samples,...])
        scores3 = np.expand_dims(scores,axis=1)
        output_shap = np.append(output_shap,scores3,axis=1)
        model_outputs.append((model.coef_, model.intercept_))
        
    return [output_shap], model_outputs


def get_major_indices(shap_scores):
    ys = [np.unique(np.nonzero(score)[1]) for score in shap_scores]
    nonzeros = [score[:, y, :] for y, score in zip(ys, shap_scores)]
    l2s = [y[np.argsort(np.sum(np.sum(nonzero ** 2, axis=2), axis=0))] for y, nonzero in zip(ys, nonzeros)]
    l1s = [y[np.argsort(np.sum(np.sum(np.abs(nonzero), axis=2), axis=0))] for y, nonzero in zip(ys, nonzeros)]
    return ys, nonzeros, l2s, l1s

def _driver(nt, dt, p, t, s):
    ds = load_datasource(nt, dt, p, t, s)
    r_x, r_y, t_x, t_y = get_batched_train_test_data(ds,p,t)
    shap_values, model_outputs = SNP_loop(r_x, t_x, r_y, num_samples=len(r_x))
    ys, nonzeros, l2s, l1s = get_major_indices(shap_values)
    ds.genome_files = None
    return {"shap_values":shap_values, "ys":ys, "nonzeros":nonzeros, "l2s":l2s, "l1s":l1s, "outputs":model_outputs, "datasource":ds}

def driver(driver_args):
    try:
        nt, dt, p, t, s = driver_args
        to_save = {"args":vars(args), "interpreted":_driver(nt, dt, p, t, s)}
        out = get_output_datapath("09_linear_regression", nt, "5_11_2021", p, t, s)
        os.makedirs(out[:out.rindex("/")], exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(to_save, f)
    except FileNotFoundError:
        print(f"Error: Could not find model or data file for args {driver_args}")



for driver_args in ((nt, dt, p, t, s) for nt in networks for dt in dates for p in phenotypes for t in thresholds for s in max_samples):
    driver(driver_args)