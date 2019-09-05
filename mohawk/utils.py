import os
from io import BufferedReader

from pkg_resources import resource_exists, resource_stream


def _ftp_path(id_, genomes_metadata):
    return genomes_metadata['ftp_path'].loc[id_]


def get_zipped_fasta_name(id_, genomes_metadata) -> str:

    return genomes_metadata['fna_gz_name'].loc[id_]


def gz_stripper(filename: str) -> str:
    if filename[-3:] == '.gz':
        return filename[:-3]
    else:
        return filename


# TODO flat directory structure
def full_fna_path(sequence_directory, id_, lineage_info):
    fasta_gz_name = get_zipped_fasta_name(id_, lineage_info)
    fasta_name = gz_stripper(fasta_gz_name)
    return os.path.join(sequence_directory, id_, fasta_name)


def _get_taxonomy(id_, lineage_info, level):
    return lineage_info[level].loc[id_]


def representative_genomes_file() -> BufferedReader:
    """Returns a buffer containing ftp links to RefSeq

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
                                  'refseq_representative_genomes_ftp.txt')


def complete_genomes_file() -> BufferedReader:
    """Returns a buffer containing ftp links to RefSeq

    Returns
    -------

    BufferedReader
        Contains the ftp links to the 'Complete' RefSeq genomes


    Raises
    ------

    IOError
        If unable to find the resource

    """

    return _fetch_resource_stream('mohawk.resources',
                                  'refseq_complete_genomes_ftp.txt')


def representative_genomes_lineage():
    return _fetch_resource_stream('mohawk.resources',
                                  'refseq_representative_genomes_lineage.txt')


def complete_genomes_lineage():
    return _fetch_resource_stream('mohawk.resources',
                                  'refseq_complete_genomes_lineage.txt')


def _fetch_resource_stream(location, name):
    if resource_exists(location, name):
        return resource_stream(location, name)
    else:
        raise IOError('Unable to find package resource {}. Please run '
                      '`setup_genome_downloads.py`.'.format((location, name)))
