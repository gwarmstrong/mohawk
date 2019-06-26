## Example commands:

```bash
$ python bin/into training-data --sequence-directory data/training_data/ --depth 10000 --length 20 --labels data/training_data/lineage.tsv --train-on genus
$ python bin/into classify --sequence-directory data/real_data
```


## Example data download:

```bash
$ ncbi-genome-download --format fasta --refseq-category reference --genus "Streptomyces coelicolor" bacteria
```

```bash
$ wget https://raw.githubusercontent.com/CAMI-challenge/CAMISIM/master/tools/assembly_summary_complete_genomes.txt
```

##
Getting ncbi taxa and filtering for bacteria

```bash
$ wget ftp://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt

$ tail -n +2 assembly_summary_refseq.txt > assembly_summary_refseq_headless.txt

```

```python
import pandas as pd
from ete3 import NCBITaxa

ncbi = NCBITaxa() # make take a few minutes if first time
assembly_info = pd.read_csv('assembly_summary_refseq.txt', sep='\t', skiprows=1, index_col=0)
taxids = assembly_info['taxid']

def safe_get_lineage(taxid):
    try:
        return ncbi.get_lineage(taxid)
    except:
        return {}

# drop non-bacteria
is_bacteria = [2 in safe_get_lineage(id) for id in taxids]
bacteria_assembly_info = assembly_info.loc[is_bacteria]

# only keep complete genomes
complete_genomes = bacteria_assembly_info['assembly_level'] == 'Complete Genome'
complete_genomes_info = bacteria_assembly_info.loc[complete_genomes]

# drop genomes excluded from refseq
included_in_refseq = complete_genomes_info['excluded_from_refseq'].isna()
included_in_refseq_info = complete_genomes_info.loc[included_in_refseq]
included_in_refseq_info.to_csv('complete_bacteria_genomes_refseq.txt', sep='\t')

# map taxonomy onto dataframe
ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

def get_lineage_array(taxid):
    lineage = ncbi.get_lineage(taxid)
    rank_dict = ncbi.get_rank(lineage)
    taxid_translator = ncbi.get_taxid_translator(lineage)
    rank_to_id = {rank: id for id, rank in rank_dict.items() if rank != 'no rank'}
    rank_ids = [rank_to_id.get(rank, 'NA') for rank in ranks] 
    return [taxid_translator[id] if id in taxid_translator else 'NA' for id in rank_ids]

lineage_info = [get_lineage_array(id) for id in included_in_refseq_info['taxid']]

lineage_df = pd.DataFrame(lineage_info, columns = ranks)
lineage_df.index = included_in_refseq_info.index
lineage_df['infraspecific_name'] = inlcuded_in_refseq_info['infraspecific_name']
included_in_refseq_info.drop('infraspecific_name', axis=1, inplace=True)
full_lineage_info = included_in_refseq_info.join(lineage_df)

full_lineage_info.to_csv('refseq_complete_genomes_taxonomy.txt', sep='\t')

slim_cols = ['taxid', 'species_taxid', 'organism_name', 'ftp_path']
slim_cols.extend(ranks)
slim_cols.append('infraspecifc_name')

full_lineage_slim = full_lineage_info[slim_cols]
full_lineage_slim.to_csv('refseq_complete_genomes_taxonomy_slim.txt', sep='\t')

    

# drop non-representative genomes
representative_categories = ['representative genome', 'reference genome']
is_representative = full_lineage_info['refseq_category'].isin(representative_categories)
representatives_info = full_lineage_info.loc[is_representative]
representatives_info.to_csv('refseq_representative_genomes_taxonomy.txt', sep='\t')
representatives_slim = representatives_info[slim_cols]
representatives_slim.to_csv('refseq_representative_genomes_taxonomy_slim.txt', sep='\t')


```

```python
from ftplib import FTP
ftp = FTP('ftp.ncbi.nih.gov')
def get_ftp_dir(abspath): return '/' + '/'.join(abspath.split('/')[3:])

```






