import os
import time
from urllib import error as urlerror
from urllib import request
import pandas as pd
from ete3 import NCBITaxa


url = 'ftp://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_'\
      'refseq.txt'
ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus',
         'species']


def safe_get_lineage(ncbi, taxid):
    try:
        return ncbi.get_lineage(taxid)
    except ValueError:
        return {}


def get_lineage_array(ncbi, taxid):
    lineage = ncbi.get_lineage(taxid)
    rank_dict = ncbi.get_rank(lineage)
    taxid_translator = ncbi.get_taxid_translator(lineage)
    rank_to_id = {rank: id_ for id_, rank in rank_dict.items() if rank !=
                  'no rank'}
    rank_ids = [rank_to_id.get(rank, 'NA') for rank in ranks]
    return [taxid_translator[id_] if id_ in taxid_translator else 'NA' for id_
            in rank_ids]


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

    ncbi = NCBITaxa()  # make take a few minutes if first time
    assembly_info = pd.read_csv(os.path.join(resources_dir,
                                             'assembly_summary_refseq.txt'),
                                sep='\t',
                                skiprows=1,
                                index_col=0)

    taxids = assembly_info['taxid']

    # drop non-bacteria
    is_bacteria = [2 in safe_get_lineage(ncbi, id) for id in taxids]
    bacteria_assembly_info = assembly_info.loc[is_bacteria]

    # only keep complete genomes
    complete_genomes = bacteria_assembly_info['assembly_level'] == 'Complete '\
                                                                   'Genome'
    complete_genomes_info = bacteria_assembly_info.loc[complete_genomes]

    # drop genomes excluded from refseq
    included_in_refseq = complete_genomes_info['excluded_from_refseq'].isna()
    included_in_refseq_info = complete_genomes_info.loc[included_in_refseq]
    # included_in_refseq_info.to_csv('complete_bacteria_genomes_refseq.txt',
    # sep='\t')

    lineage_info = [get_lineage_array(ncbi, id_) for id_ in
                    included_in_refseq_info['taxid']]

    lineage_df = pd.DataFrame(lineage_info, columns=ranks)
    lineage_df.index = included_in_refseq_info.index
    lineage_df['infraspecific_name'] = included_in_refseq_info[
        'infraspecific_name']
    included_in_refseq_info.drop('infraspecific_name', axis=1, inplace=True)
    full_lineage_info = included_in_refseq_info.join(lineage_df)

    # full_lineage_info.to_csv('refseq_complete_genomes_taxonomy.txt',
    #                          sep='\t')

    full_lineage_info['fna_gz_name'] = [ftp_dir.split('/')[-1] +
                                        '_genomic.fna.gz' for ftp_dir in
                                        full_lineage_info['ftp_path']]

    # print(full_lineage_info.head())

    lineage_cols = ['taxid', 'fna_gz_name', 'organism_name']
    lineage_cols.extend(ranks)
    lineage_cols.append('infraspecific_name')

    ftp_cols = ['ftp_path', 'fna_gz_name']

    # split into `ftp_links` and `lineage`
    complete_lineage = full_lineage_info[lineage_cols]
    complete_lineage.to_csv(resource('refseq_complete_genomes_lineage.txt'),
                            sep='\t', na_rep='na')

    complete_ftp = full_lineage_info[ftp_cols]
    complete_ftp.to_csv(resource('refseq_complete_genomes_ftp.txt'),
                        sep='\t', na_rep='na')

    # drop non-representative genomes
    representative_categories = ['representative genome', 'reference genome']
    is_representative = full_lineage_info['refseq_category'].isin(
         representative_categories)

    representatives_info = full_lineage_info.loc[is_representative]
    # representatives_info.to_csv('refseq_representative_genomes_taxonomy.txt',
    # sep='\t')
    representatives_lineage = representatives_info[lineage_cols]
    representatives_lineage.to_csv(
            resource('refseq_representative_genomes_lineage.txt'),
            sep='\t', na_rep='na')

    representatives_ftp = representatives_info[ftp_cols]
    representatives_ftp.to_csv(
        resource('refseq_representative_genomes_ftp.txt'), sep='\t',
        na_rep='na')


if __name__ == "__main__":
    setup_genome_downloads()
