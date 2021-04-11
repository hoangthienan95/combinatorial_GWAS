# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/package/phenotypes.ipynb (unless otherwise specified).

__all__ = ['QueryDataframe', 'parameters', 'catalog_all', 'catalog_all', 'read_csv_compressed', 'get_GWAS_result_link',
           'icd10_pheno_matrix', 'icd10_primary_cols', 'icd10_pheno_matrix', 'get_phenotype', 'get_GWAS_snps_for_trait']

# Cell
from .data_catalog import get_catalog, get_parameters
import combinatorial_gwas
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from functools import partial
import numpy as np
from typing import List, Union
from fastcore.utils import partialler
import logging

# Cell

@pd.api.extensions.register_dataframe_accessor("pheno")
@dataclass
class QueryDataframe():
    df: pd.DataFrame

    def query(self, **column_dict:dict):
        query_str = " and ".join([f"({col} {cond})" for col, cond in column_dict.items()])
        return self.df.query(query_str)

# Cell
#hide_output

parameters = get_parameters()
parameters

# Cell
catalog_all = get_catalog()
catalog_all = catalog_all.reload()
catalog_all.list()

# Cell
read_csv_compressed= partialler(pd.read_csv, sep="\t", compression= "gzip")
get_GWAS_result_link = partialler(parameters['template_gwas_result_file_link'].format)


# Cell
icd10_pheno_matrix = catalog_all.load("ICD10_pheno_matrix")

#get the first 3 character of ICD code
icd10_primary_cols = icd10_pheno_matrix.columns[icd10_pheno_matrix.columns.str.contains("primary")]
icd10_pheno_matrix = icd10_pheno_matrix.astype(str).apply(lambda x: x.str.slice(0,3))



# Cell
def get_phenotype(icd10_codes: Union[str, List[str]] ="I84"):
    icd10_codes = [icd10_codes] if not isinstance(icd10_codes, list) else icd10_codes
    pheno_df_list = [icd10_pheno_matrix[icd10_primary_cols].isin([icd10_code]).any(axis=1).astype(int) for icd10_code in icd10_codes]
    pheno_df = pd.concat(pheno_df_list, axis=1)
    pheno_df.columns = icd10_codes
    return pheno_df

# Cell

def get_GWAS_snps_for_trait(phenotype_code= "I84", chromosome:int = 21, sort_val_cols_list: List[str] = ["pval"], ascending_bool_list: List[bool] = [False], id_only= True):
    chromosome_str = f"{chromosome}:"
    gwas_result_df = read_csv_compressed(get_GWAS_result_link(phenotype_code=phenotype_code)).query(f"variant.str.startswith('{chromosome_str}')")
    gwas_result_df = gwas_result_df.reset_index(drop=True).reset_index().rename(columns = {"index":"position_rank"})
    gwas_result_df = gwas_result_df.sort_values(sort_val_cols_list, ascending = ascending_bool_list)
    variant_id_df = gwas_result_df["variant"].str.split(":",expand=True)
    variant_id_df["chr1_4"] =variant_id_df[[1,2,3]].apply("_".join, axis=1)
    variant_id_df[1] = variant_id_df[1].astype(int)
    gwas_result_df[["chr", "position", "major_allele"]] = variant_id_df[[0, 1, 2]]
    gwas_result_df["full_id"] =  variant_id_df[[0, "chr1_4"]].apply(":".join, axis=1)

    if id_only:
        return gwas_result_df["full_id"].values
    else:
        return gwas_result_df