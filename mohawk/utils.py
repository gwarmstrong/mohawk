import os


def _ftp_path(id_, genomes_metadata):
    return genomes_metadata['ftp_path'].loc[id_]


def get_fna_name(id_, genomes_metadata) -> str:

    return genomes_metadata['fna_gz_name'].loc[id_]


def gz_stripper(filename: str) -> str:
    if filename[-3:] == '.gz':
        return filename[:-3]
    else:
        return filename


def full_fna_path(sequence_directory, id_, lineage_info):
    fna_gz_name = get_fna_name(id_, lineage_info)
    fna_name = gz_stripper(fna_gz_name)
    return os.path.join(sequence_directory, id_, fna_name)


def _get_taxonomy(id_, lineage_info, level):
    return lineage_info[level].loc[id_]


