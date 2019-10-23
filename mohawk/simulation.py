import numpy as np
import skbio
from typing import List, Optional

from mohawk._format import sample_from_contig_set


# TODO flat directory structure for sequence_directory
def simulate_from_genomes(distribution: List[float], total_reads: int,
                          length: int, file_list: List[str],
                          sequence_directory: Optional[str] = None,
                          random_seed: Optional[int] = None):

    if random_seed is not None:
        np.random.seed(random_seed)
    random_func = np.random.randint

    if len(file_list) != len(distribution):
        raise ValueError("id_list and distribution must have the same shape")

    # use multinomial to get number of reads for each id
    id_depths = [round(val * total_reads) for val in distribution]

    # then simulate reads from each
    all_reads = []
    for idx, (file_, depth) in enumerate(zip(file_list, id_depths)):
        if depth > 0:
            sequences = list(skbio.io.read(file_, format='fasta'))
            reads = sample_from_contig_set(sequences, depth, length,
                                           random_func)
            all_reads.append(reads)

    reads = np.vstack(all_reads)

    # TODO comment this with helpful comment
    ids = [id_ for id_, depth in zip(file_list, id_depths) for _ in
           range(depth)]

    return reads, ids
