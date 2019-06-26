import warnings
import hashlib
from pkg_resources import resource_stream

assembly_hash_exp = '3e4c72a803afcc28fd1016904b09da28'

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
