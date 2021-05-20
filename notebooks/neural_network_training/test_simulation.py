from combinatorial_gwas.simulation import SimulatedPheno, SNPInfoUnit
from combinatorial_gwas.high_level import datasource

test_datasource = chromosome_datasource(chromosomes = list(range(1, 23)), snp_filters= [snp_filter('I83', snp_filter.SORT_PVALUE, threshold= 1e-6)], max_samples=100_000, balance_pheno="I83")

