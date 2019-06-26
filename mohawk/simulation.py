import os
import numpy as np
import pandas as pd
import skbio
from pkg_resources import resource_stream
from typing import List, Optional

from mohawk.utils import full_fna_path, _get_taxonomy
from mohawk._format import sample_from_contig_set

representative_genomes_lineage = resource_stream(
    'mohawk.resources', 'refseq_representative_genomes_lineage.txt'
)

complete_genomes_lineage = resource_stream(
    'mohawk.resources', 'refseq_complete_genomes_lineage.txt')


def simulate_from_genomes(id_list: List[str],
                          distribution: List[float],
                          total_reads: int,
                          length: int,
                          channel: Optional[str] = 'representative',
                          sequence_directory: Optional[str] = None,
                          random_seed: Optional[int] = None):

    if channel == 'representative':
        lineage_info = pd.read_csv(representative_genomes_lineage,
                                   sep='\t', index_col=0)
    elif channel == 'complete':
        lineage_info = pd.read_csv(complete_genomes_lineage, sep='\t',
                                   index_col=0)
    else:
        raise ValueError("Invalid choice for `channel`. Options are "
                         "'representative' and 'complete'.")
    if random_seed is not None:
        np.random.seed(random_seed)
    random_func = np.random.randint

    if len(id_list) != len(distribution):
        raise ValueError("id_list and distribution must have the same shape")

    if sequence_directory is None:
        sequence_directory = os.path.curdir()

    # use multinomial to get number of reads for each id
    id_depths = np.random.multinomial(total_reads, distribution)
    print('id_depths: {}'.format(id_depths))

    # then simulate reads from each
    all_reads = []
    for idx, (id_, depth) in enumerate(zip(id_list, id_depths)):
        if depth > 0:
            sequence_file = full_fna_path(sequence_directory, id_,
                                          lineage_info)
            sequences = list(skbio.io.read(sequence_file, format='fasta'))
            reads = sample_from_contig_set(sequences, depth, length,
                                           random_func)
            all_reads.append(reads)

    reads = np.vstack(all_reads)

    ids = [id_ for id_, depth in zip(id_list, id_depths) for _ in
           range(depth)]

    return reads, ids


def id_to_lineage(ids: List[str],
                  level: str,
                  channel: Optional[str] = 'representative') -> List[str]:
    """

    Parameters
    ----------
    ids
        List of assembly accession ids
    level
        Taxonomic level to use for class labels
        Options: ['superkingdom', 'phylum', 'class', 'order', 'family',
          'genus', 'species']
    channel
        The set of assembly accessions that should be checked for the
          ids in `ids`

    Returns
    -------
    List[str]
        Each entry is the taxonomic classification at `level` for
          each id

    """

    if channel == 'representative':
        lineage_info = pd.read_csv(representative_genomes_lineage,
                                   sep='\t', index_col=0)
    elif channel == 'complete':
        lineage_info = pd.read_csv(complete_genomes_lineage, sep='\t',
                                   index_col=0)
    else:
        raise ValueError("Invalid choice for `channel`. Options are "
                         "'representative' and 'complete'.")

    return [_get_taxonomy(id_, lineage_info, level) for id_ in ids]


