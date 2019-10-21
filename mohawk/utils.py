import os
from io import BufferedReader

from pkg_resources import resource_exists, resource_stream


def _ftp_path(id_, genomes_metadata):
    return genomes_metadata['ftp_path'].loc[id_]


def get_zipped_fasta_name(id_, genomes_metadata) -> str:
    return genomes_metadata['ftp_path'].loc[id_].split('/')[-1] + \
           '_genomic.fna.gz'


def gz_stripper(filename: str) -> str:
    if filename[-3:] == '.gz':
        return filename[:-3]
    else:
        return filename


# TODO flat directory structure
def full_fna_path(sequence_directory, id_, metadata):
    fasta_gz_name = get_zipped_fasta_name(id_, metadata)
    fasta_name = gz_stripper(fasta_gz_name)
    return os.path.join(sequence_directory, id_, fasta_name)


def default_metadata() -> BufferedReader:
    """"Returns a buffer containing the NCBI metadata file included with
    this package

    Returns
    -------

    BufferedReader
        Contains the ftp links to the 'Representative' RefSeq genomes


    Raises
    ------

    IOError
        If unable to find the resource

    """
    return _fetch_resource_stream('mohawk.resources',
                                  'assembly_summary_refseq.txt')


def _fetch_resource_stream(location, name):
    if resource_exists(location, name):
        return resource_stream(location, name)
    else:
        raise IOError('Unable to find package resource {}'
                      .format((location, name)))
