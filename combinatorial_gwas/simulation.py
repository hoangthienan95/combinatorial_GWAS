# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/package/simulation.ipynb (unless otherwise specified).

__all__ = ['SNPInfoUnit', 'SimulatedPheno', 'get_chrom_to_variants_dict', 'get_simulation_geno_df', 'get_genotype_dict',
           'get_two_snps_queries']

# Cell
from .genotype import load_genetic_file
from .phenotypes import get_phenotype, read_csv_compressed
from .data_catalog import get_parameters, get_catalog
import pandas as pd
from dataclasses import dataclass
from functools import partial, lru_cache
from collections import defaultdict
from fastcore.utils import partialler
import operator
from itertools import product, chain
import numpy as np
#import apricot
from sklearn import preprocessing
from typing import List
from tqdm.auto import tqdm

# Cell
@dataclass
class SNPInfoUnit:
    negation: str
    snp_id: str
    geno: str

# Cell
@dataclass
class SimulatedPheno:
    snps: SNPInfoUnit
    op: str
    query: str
    pheno_col: np.array
    case_count: int
    control_count: int



# Cell
def get_chrom_to_variants_dict(pairs_df):
    snp_set = set(pairs_df["snp1"].tolist() + pairs_df["snp2"].tolist())
    chrom_to_variants_dict = defaultdict(list)
    for snp_id in snp_set:
        chrom = int(snp_id.split(":")[0])
        chrom_to_variants_dict[chrom].append(snp_id)
    return chrom_to_variants_dict


# Cell
def get_simulation_geno_df(datasource, chrom_to_variants_dict, sample_ids=None):
    all_genos_all_chrom = []
    all_variants = []
    if sample_ids == None:
        sample_ids = datasource.samples
    for chrom, variants in chrom_to_variants_dict.items():
        genetic_file = datasource.genome_files[chrom]
        variant_arr = genetic_file.get_geno_each_sample(sample_ids = sample_ids, variant_ids = variants, one_hot_encoded=False)
        all_genos_all_chrom.append(variant_arr)
        all_variants += variants
    all_genos_all_chrom = np.hstack(all_genos_all_chrom)
    geno_df = pd.DataFrame(all_genos_all_chrom, columns = all_variants, index = sample_ids)
    return geno_df



# Cell
def get_genotype_dict(geno_df):
    geno_dict = {col: [SNPInfoUnit(*item) for item in product(["not", ""],[col], geno_df[col].unique())] for col in geno_df.columns}
    return geno_dict

# Cell
def get_two_snps_queries(geno_dict:dict, two_snp_rsid_list:List[str], geno_df):
    print(f"CREATING SIMULATED PHENOTYPES FOR SNP PAIRS {two_snp_rsid_list}")
    all_query_dict= {}
    two_snp_list = list(product(geno_dict[two_snp_rsid_list[0]], ["and", "or"], geno_dict[two_snp_rsid_list[1]])) + [[single_snp_query] for single_snp_query in geno_dict[two_snp_rsid_list[0]]] + [[single_snp_query] for single_snp_query in geno_dict[two_snp_rsid_list[1]]]
    snp_unit = "({negation} (`{snp_id}` == {geno}))"
    nan_snp_unit = "({negation} (`{snp_id}`.isna()))"

    for ele in two_snp_list:
        query = ""
        snps = []
        ops = []
        has_nan = None
        for part in ele:
            if isinstance(part, SNPInfoUnit):
                if np.isnan(part.geno):
                    unit = nan_snp_unit.format(**vars(part))
                    has_nan = True
                else:
                    unit = snp_unit.format(**vars(part))

                snps.append(part)
            else:
                unit = part
                ops.append(unit)
            query += " "
            query += unit
        if has_nan == True:
            continue
        pheno_col = geno_df.eval(query).astype(int)
        try:
            case_count = pheno_col.value_counts()[1]
        except KeyError:
            case_count = None
        try:
            control_count = pheno_col.value_counts()[0]
        except KeyError:
            control_count = None
        simulated_pheno_obj = SimulatedPheno(query = query, snps = snps, op = ops, pheno_col = pheno_col, case_count = case_count, control_count = control_count)
        all_query_dict[query] = simulated_pheno_obj

    return all_query_dict