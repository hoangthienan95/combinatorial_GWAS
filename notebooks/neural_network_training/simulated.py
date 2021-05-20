import argparse

parser = argparse.ArgumentParser(description='Tune and train a model')
# Use multiprocessing to call get_data in parallel
parser.add_argument('--pool-data-acquisition', '-mpdata', action='store_true')
# Turns out Tensorflow doesn't play well with multiprocessing; for now, phenotypes should only be length 1
# THIS WILL NOT WORK IF YOU PASS MULTIPLE PHENOTYPES AND POOL_DATA_ACQUISITION
parser.add_argument('--phenotypes', nargs='+')
parser.add_argument('--thresholds', '-t', type=float, nargs='+', required=True)
# Only use DNN
parser.add_argument('--network', '-n', choices=['DNN', 'dnn'], required=True)
parser.add_argument('--pair', nargs=2, required=True)
parser.add_argument('--exp-id', '-i', default='')
parser.add_argument('--batch-size', '-b', type=int, default=1000)
parser.add_argument('--data-batch', '-db', type=int, default=50000)
parser.add_argument('--epochs', '-e', type=int, default=20)
parser.add_argument('--max-samples', '-m', type=int, default=100_000)
parser.add_argument('--random-state', '-r', type=int, default=42)
parser.add_argument('--validation-split', '-v', type=float, default=0.2)
parser.add_argument('--test-split', type=float, default=0.1)
parser.add_argument('--tune-split', type=float, default=0.01)
parser.add_argument('--optimize-split', type=float, default=0.2)
# Use multiprocessing to speed up Tensorflow computations
parser.add_argument('--use-multiprocessing', '-mptf', action='store_true')
parser.add_argument('--create-test-output', action='store_true')
parser.add_argument('--model-directory', '-mdir', type=str, default='/lab/corradin_biobank/FOR_AN/combinatorial_GWAS/data/06_models/{}/5_11_2021')
parser.add_argument('--test-directory', '-tdir', type=str, default='/lab/corradin_biobank/FOR_AN/combinatorial_GWAS/data/07_model_output/{}/5_11_2021')
parser.add_argument('--quiet', '-q', action='store_true')
parser.add_argument('--query-index', type=int, default=None)


args = parser.parse_args()

globals().update(vars(args))

if not all(0 <= t <= 1 for t in thresholds):
    raise ValueError(f"Thresholds must be between 0 and 1! (Got thresholds of {thresholds})")
    
network = network.lower()
if '{}' in model_directory:
    model_directory = model_directory.replace('{}', network)
if '{}' in test_directory:
    test_directory = test_directory.replace('{}', network)

if exp_id == '':
    exp_id = f'simulated_phenotypes_{"_".join(phenotypes)}_thresholds_{"_".join(str(t) for t in thresholds)}_max_samples_{max_samples}_network_{network}'

if not quiet:
    print('Options:')
    print('\n'.join(f'{k}: {v}' for k,v in vars(args).items()))

import sys
sys.path.append('../../')
from combinatorial_gwas.simulation import SimulatedPheno, SNPInfoUnit
import combinatorial_gwas.high_level as cgwas
import tensorflow as tf
import kerastuner as kt
import os
import shutil
import string
import sklearn
import math
import pickle

def make_DNN_model(d1, dropout, l2, A, P):
    regularizer = tf.keras.regularizers.L2(l2)
    return tf.keras.Sequential(layers=[
        tf.keras.layers.InputLayer(input_shape=(A, 4)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(d1, activation=tf.nn.relu, kernel_regularizer=regularizer),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(P, activation=tf.nn.sigmoid, kernel_regularizer=regularizer)
    ], name='dnn_model')

def compile_model(model, optimizer):
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', tf.metrics.AUC(name='AuROC'), tf.metrics.AUC(name='AuPR', curve='PR')])
    model.summary()

class Range:
  def __init__(self, min, max, step=None, sampling='linear'):
    self.min = min
    self.max = max
    self.step = step
    self.sampling = sampling
  
  def to_Int(self, hp, name):
    return hp.Int(name, min_value = self.min, max_value = self.max, step = self.step)
  
  def to_Float(self, hp, name):
    return hp.Float(name, min_value = self.min, max_value=self.max, step=self.step)

l2_choices = [0.5, 0.2, 0.1, 0.08, 0.05, 0.01]
lr_choices = [0.1, 0.05, 0.01, 0.005, 0.001]
dropout_choices = [0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.65, 0.8]

def load_pickled_hyperparameters(file):
    with open(file, 'rb') as f:
        out = pickle.load(f)
        return out["outputs"]["hyperparameters"].values

def get_tunable_DNN_model(hp, d1, A, P):
    d1_units = d1.to_Int(hp, 'd1_units')
    dropout = hp.Choice('dropout', values=dropout_choices)
    l2 = hp.Choice('l2', values=l2_choices)
    lr = hp.Choice('lr', values=lr_choices)
    model = make_DNN_model(d1_units, dropout, l2, A, P)
    compile_model(model, tf.keras.optimizers.Adam(lr))
    return model

def get_tuner(model_builder):
    global exp_id
    try:
      shutil.rmtree(f'hp_tuning_{exp_id}')
    except FileNotFoundError:
      pass
    return kt.Hyperband(model_builder, objective=kt.Objective('val_AuPR', direction='max'), max_epochs=epochs, executions_per_trial=3, directory=f'hp_tuning_{exp_id}', project_name='initial_model')

def get_datasource(phenotype, threshold):
    datasource = cgwas.chromosome_datasource(snp_filters=[cgwas.snp_filter(phenotype, cgwas.snp_filter.SORT_PVALUE, threshold=threshold)], balance_pheno=phenotype, max_samples=max_samples, random_state=random_state, data_split_params=cgwas.DataSplitParams(validation_split=validation_split, test_split=test_split, tune_split=tune_split, validation_tune_split = optimize_split))
    return datasource

def get_tunable_hyperparameters_DNN(search_train_x):
    A = search_train_x.shape[1]
    P = 1
    d1 = Range(max(P // 2, 1), P * 128, 16)
    return d1, A, P

def tuned_model(nt, dt, p, t, s, search_train_x):
#     tuner = get_tuner(lambda hp: get_tunable_DNN_model(hp, *get_tunable_hyperparameters_DNN(search_train_x)))
#     tuner.search_space_summary()
#     tuner.search(search_train_x, search_train_y, epochs=epochs, validation_data=(search_validation_x, search_validation_y))
#     tuner.results_summary()
    filename = f"../../data/07_model_output/{nt}/{dt}/phenotype_{p}_threshold_{t}_max-samples_{s}"
    hp = load_pickled_hyperparameters(filename)
    model = make_DNN_model(hp['d1_units'], hp['dropout'], hp['l2'], search_train_x.shape[1], 1)
    compile_model(model, tf.keras.optimizers.Adam(hp['lr']))
    return model, None, hp

def train_test_model(model, train_x, train_y, validation_x, validation_y, test_x, test_y):
    history = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, validation_data=(validation_x, validation_y), use_multiprocessing=use_multiprocessing, workers=os.cpu_count() - 1 if use_multiprocessing else 1)
    return history, model.evaluate(x=test_x, y=test_y, use_multiprocessing=use_multiprocessing, workers=os.cpu_count() - 1 if use_multiprocessing else 1)

save_model = lambda model, filename: model.save(f'{model_directory}/{filename}')
    
def dataframe_output(model_output, output_names):
    import pandas as pd
    return pd.DataFrame.from_dict({name:[out] for name, out in zip(output_names, model_output)})
    
def save_data(outputs, datasource, filename):
    import pickle
    os.makedirs(test_directory, exist_ok=True)
    to_save = {}
    to_save['outputs'] = outputs
    to_save['args'] = vars(args)
    genome_files = datasource.genome_files
    datasource.genome_files = None
    to_save['datasource'] = datasource
    with open(f'{test_directory}/{filename}', 'wb') as f:
        pickle.dump(to_save, f)
    datasource.genome_files = genome_files

def _driver(phenotype, threshold, tuning_data, train_validation_test_data, datasource, query):
    tune_x, optimize_x, tune_y, optimize_y = tuning_data
    train_x, train_y, validation_x, validation_y, test_x, test_y = train_validation_test_data
    model, tuner, hyperparameters = tuned_model(network, '4_29_2021', phenotype, threshold, max_samples, test_x)
    history, test_output = train_test_model(model, *train_validation_test_data)
    evaluated = {}
    evaluated['test'] = test_output
    evaluated['train'] = model.evaluate(x=train_x, y=train_y, use_multiprocessing=use_multiprocessing, workers=os.cpu_count() - 1 if use_multiprocessing else 1)
    evaluated['validation'] = model.evaluate(x=validation_x, y=validation_y, use_multiprocessing=use_multiprocessing, workers=os.cpu_count() - 1 if use_multiprocessing else 1)
#     evaluated['tune'] = model.evaluate(x=tune_x, y=tune_y, use_multiprocessing=use_multiprocessing, workers=os.cpu_count() - 1 if use_multiprocessing else 1)
#     evaluated['optimize'] = model.evaluate(x=optimize_x, y=optimize_y, use_multiprocessing=use_multiprocessing, workers=os.cpu_count() - 1 if use_multiprocessing else 1)
    evaluated = {k:dataframe_output(v, model.metrics_names) for k, v in evaluated.items()}
    evaluated['history'] = history.history
    evaluated['hyperparameters'] = hyperparameters
    filename = f'simulated_phenotype_{phenotype}_threshold_{threshold}_max-samples_{max_samples}_query_{query}'
    save_model(model, filename)
    if create_test_output:
        save_data(evaluated, datasource, filename)
    for i in range(len(train_x)):
        del train_x[i]

def get_batched_tune_train_validation_test_data(datasource, phenotype, threshold):
    r_x, r_y = batched_train_data(datasource, phenotype, threshold, 'train')
    v_x, v_y = datasource.get_simulated_data(slice(0, None), 'validation', pair)
    t_x, t_y = datasource.get_simulated_data(slice(0, None), 'test', pair)
    rt_x, rt_y = None, None# datasource.get_simulated_data(slice(0, None), 'train_tune', pair)
    vt_x, vt_y = None, None#datasource.get_simulated_data(slice(0, None), 'validation_tune', pair)
    return rt_x, vt_x, rt_y, vt_y, r_x, r_y, v_x, v_y, t_x, t_y

def acquire_data(args):
    phenotype, threshold = args
    datasource = get_datasource(phenotype, threshold)
    data = get_batched_tune_train_validation_test_data(datasource, phenotype, threshold)
    return (*data, datasource)

queries = sorted(cgwas.simulation_I83_queries_pheno_dict[tuple(pair)])

if query_index != None:
    queries = [queries[query_index]]

def batched_train_data(datasource, phenotype, threshold, dataset):
    class TrainingData(tf.keras.utils.Sequence):
        def __init__(self, datasource, phenotype, threshold, dataset, query):
            self.datasource = datasource
            self.phenotype = phenotype
            self.threshold = threshold
            self.dataset = dataset
            self.batch_indeces = list(range(0, datasource.max_samples, data_batch))
            if self.batch_indeces[-1] != datasource.max_samples:
                self.batch_indeces.append(datasource.max_samples)
            self.saved = [f'{test_directory}/train_cache/{f"phenotype_{self.phenotype}_threshold_{self.threshold}_max_samples_{max_samples}_dataset_{self.dataset}_{idx}_X.cache"}' for idx in range(len(self))]
            self.query = query
            try:
                self.y_df = cgwas.simulation_I83_queries_pheno_dict[tuple(pair)][self.query]
            except KeyError:
                self.y_df = cgwas.simulated_I83_queries_pheno_dict[tuple(reversed(pair))][self.query]
            print(type(datasource))
        
        def __len__(self):
            return len(self.batch_indeces) - 1
        
        def __getitem__(self, idx):
            import pickle
            os.makedirs(f'{test_directory}/train_cache', exist_ok=True)
            x_filename = self.saved[idx]
            s = slice(*self.batch_indeces[idx:idx+2])
            try:
                with open(x_filename, 'rb') as f:
                    x = pickle.load(f)
            except (FileNotFoundError, EOFError) as e:
                x, y = self.datasource.get_simulated_data(s, self.dataset, pair)
                y = y[self.query].values
                with open(x_filename, 'wb') as f:
                    pickle.dump(x, f)
                return x, y
            print(type(self.datasource))
            _, sample_id_subset = self.datasource.get_sample_id_in_split(s, self.dataset)
            y = self.y_df.pheno_col.loc[sample_id_subset].values
            return x, y
        
        def __delitem__(self, idx):
            pass
        
        def clean():
            for idx in range(len(self)):
                try:
                    os.remove(self.saved[idx])
                except:
                    pass
    return {query:TrainingData(datasource, phenotype, threshold, dataset, query) for query in queries}, None

def driver():
    for p in phenotypes:
        for threshold in thresholds:
            rt_x, vt_x, rt_y, vt_y, r_x, r_y, v_x, v_y, t_x, t_y, datasource = acquire_data((p, threshold))
            for query in queries:
                _driver(p, threshold, (rt_x, vt_x, rt_y, vt_y), (r_x[query], r_y, v_x, v_y[query].values, t_x, t_y[query].values), datasource, query)

driver()