from .genotype import load_genetic_file
from .data_catalog import get_catalog, get_config, get_parameters
from .phenotypes import get_phenotype, get_GWAS_snps_for_trait
from typing import List, Union
import numpy as np

class snp_filter:
    SORT_PVALUE = 'pval'
    SORT_BETA = 'beta'
    def __init__(self, phenotype:str, sort:str, threshold=1e-5):
        self.phenotype = phenotype
        self.sort = sort
        self.threshold = threshold

class chromosome_datasource:
    def __init__(self, chromosomes=list(range(1, 23))):
        self.genome_file = load_genetic_file(max(chromosomes))
        self.chromosomes = chromosomes
    
    def get_data(self, snp_filters=[snp_filter('I84', snp_filter.SORT_PVALUE)], samples=None, max_samples=float('inf')):
        
        if samples is None:
            samples = self.genome_file.samples
        sample_index, pheno_df_ordered = get_phenotype([snp.phenotype for snp in snp_filters], samples = samples)
        
        variants_indeces = []
        for snp in snp_filters:
            subset_snps = get_GWAS_snps_for_trait(snp.phenotype, chromosome=self.chromosomes, id_only=False, sort_val_cols_list=snp.sort, ascending_bool_list=[snp.sort == snp_filter.SORT_PVALUE]).query(f'{snp.sort} {"<" if snp.sort == snp_filter.SORT_PVALUE else ">"} {snp.threshold}').sort_values('position')['full_id'].values
            #sorter = np.argsort(self.genome_file.bgen_reader_obj.ids)
            variants_index = self.genome_file.get_variant_index(subset_snps) #sorter[np.searchsorted(self.genome_file.bgen_reader_obj.ids, subset_snps, sorter=sorter)]
            variants_indeces.append(variants_index)
        variants_index = np.unique(np.concatenate(variants_indeces))
        
        probs = self.genome_file.bgen_reader_obj.read((sample_index, variants_index))
        ohe_genetic_info = np.identity(3)[self.genome_file.get_geno_each_sample(probs,"max").astype(int)] #sometimes it has Nans so need to convert to type int
        if max_samples == float('inf'):
            return ohe_genetic_info, pheno_df_ordered.iloc[:, :]
        else:
            return ohe_genetic_info[:max_samples, :], pheno_df_ordered.iloc[:max_samples, :]
        