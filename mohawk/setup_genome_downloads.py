import os
import time
from urllib import error as urlerror
from urllib import request
import pandas as pd


url = 'ftp://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_'\
      'refseq.txt'
ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus',
         'species']


def setup_genome_downloads(base_dir=None, urlretrieve_tries=10):
    if base_dir is None:
        base_dir = os.path.curdir()
    resources_dir = os.path.join(base_dir, 'resources')

    def resource(file_name_): return os.path.join(resources_dir, file_name_)

    file_name = os.path.join(resources_dir, 'assembly_summary_refseq.txt')
    count = 0
    result = False
    while (count < urlretrieve_tries) and (not result):
        try:
            request.urlretrieve(url, file_name)
        except urlerror.URLError:
            print("The command `request.urlretrieve(url, file_name)` failed. "
                  "Retrying, {} of {}".format(count, urlretrieve_tries))
            time.sleep(10)
            count += 1
        else:
            result = True

    if not result:
        raise urlerror.URLError("The command `request.urlretrieve(url, "
                                "file_name)` failed {} times.".format(
                                    urlretrieve_tries)
                                )

    assembly_info = pd.read_csv(os.path.join(resources_dir,
                                             'assembly_summary_refseq.txt'),
                                sep='\t',
                                skiprows=1,
                                index_col=0)

    # only keep complete genomes
    complete_genomes = assembly_info['assembly_level'] == 'Complete Genome'
    complete_genomes_info = assembly_info.loc[complete_genomes]

    # drop genomes excluded from refseq
    included_in_refseq = complete_genomes_info['excluded_from_refseq'].isna()
    included_in_refseq_info = complete_genomes_info.loc[included_in_refseq]

    full_lineage_info = included_in_refseq_info
    full_lineage_info['fna_gz_name'] = [ftp_dir.split('/')[-1] +
                                        '_genomic.fna.gz' for ftp_dir in
                                        full_lineage_info['ftp_path']]

    ftp_cols = ['ftp_path', 'fna_gz_name']

    complete_ftp = full_lineage_info[ftp_cols]
    complete_ftp.to_csv(resource('refseq_complete_genomes_ftp.txt'),
                        sep='\t', na_rep='na')

    # drop non-representative genomes
    representative_categories = ['representative genome', 'reference genome']
    is_representative = full_lineage_info['refseq_category'].isin(
         representative_categories)

    representatives_info = full_lineage_info.loc[is_representative]

    representatives_ftp = representatives_info[ftp_cols]
    representatives_ftp.to_csv(
        resource('refseq_representative_genomes_ftp.txt'), sep='\t',
        na_rep='na')


if __name__ == "__main__":
    setup_genome_downloads()
