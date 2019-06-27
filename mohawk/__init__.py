import os
import warnings
import hashlib
from pkg_resources import resource_stream, resource_exists
from mohawk.setup_genome_downloads import setup_genome_downloads

assembly_hash_exp = '3e4c72a803afcc28fd1016904b09da28'

all_resources = ['assembly_summary_refseq.txt',
                 'refseq_complete_genomes_ftp.txt',
                 'refseq_complete_genomes_lineage.txt',
                 'refseq_representative_genomes_ftp.txt',
                 'refseq_representative_genomes_lineage.txt'
                ]


if not all(resource_exists('mohawk.resources', res) for res in all_resources):
    print("Downloading NCBI metadata... This may take a moment.")
    init_file = __file__
    init_dir = os.path.join(*(os.path.split(init_file)[:-1]))
    setup_genome_downloads(base_dir=init_dir)

# if the resources do not exists, run setup_genome_downloads

with resource_stream('mohawk.resources', 'assembly_summary_refseq.txt') as fp:
    assembly_hash_obs = hashlib.md5(fp.read()).hexdigest()

# TODO this functionality will probably be annoying since the ncbi database
#  will probably be update multiple times a day, should only warn if
#  the format of the assembly has changes -> do check for this instead.
#  Maybe do something like check that the right columns exist
if assembly_hash_exp != assembly_hash_obs:
    hash_msg = "Assembly Summary has been updated."
    warnings.warn(hash_msg, ImportWarning)
    print("WARNING: Assembly Summary has been updated: {}".format(
        assembly_hash_obs))
