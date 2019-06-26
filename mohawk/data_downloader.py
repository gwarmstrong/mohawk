import os
import gzip
import shutil
import pandas as pd
from ftplib import FTP
from typing import List, Optional
from pkg_resources import resource_stream
from mohawk.utils import _ftp_path, get_fna_name, gz_stripper

representative_genomes_file = resource_stream(
        'mohawk.resources', 'refseq_representative_genomes_ftp.txt'
        )

complete_genomes_file = resource_stream(
        'mohawk.resources', 'refseq_complete_genomes_ftp.txt')


def _ncbi_ftp_downloader(id_list: List[str],
                         genomes_metadata: pd.DataFrame,
                         genomes_directory: str) -> bool:

    ftp = FTP('ftp.ncbi.nih.gov')
    ftp.login(user='anonymous', passwd='example@net.com')
    for id_ in id_list:
        abspath = _ftp_path(id_, genomes_metadata)
        ftp_dir = get_ftp_dir(abspath)
        ftp.cwd(ftp_dir)
        filename = get_fna_name(id_, genomes_metadata)
        local_dir = os.path.join(genomes_directory, id_, filename)
        ftp.retrbinary("RETR " + filename, open(local_dir, 'wb').write)

    return True


def _gunzip(gz_file: str, gunzipped_file: str) -> bool:
    # see 'https://stackoverflow.com/questions/48466421/python-how-to-'
    #     'decompress-a-gzip-file-to-an-uncompressed-file-on-disk'
    with gzip.open(gz_file, 'r') as f_in, \
            open(gunzipped_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    return True


def _file_gunzipper(id_list: List[str],
                    genomes_metadata: pd.DataFrame,
                    genomes_directory: str) -> None:

    for id_ in id_list:
        filename = get_fna_name(id_, genomes_metadata)
        fna_gz_filename = os.path.join(genomes_directory, id_, filename)
        fna_filename = gz_stripper(fna_gz_filename)
        _gunzip(fna_gz_filename, fna_filename)


def get_ftp_dir(abspath: str) -> str:
    """
    Returns path within ftp site

    Parameters
    ----------
    abspath
        The full ftp url

    Returns
    -------
    str
        The relative path within the ftp site

    Examples
    --------
    >>> abspath = 'ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/900/128/'\
                  '725/GCF_900128725.1_BCifornacula_v1.0'
    >>> get_ftp_dir(abspath)
    'genomes/all/GCF/900/128/725/GCF_900128725.1_BCifornacula_v1.0'

    """
    return '/' + '/'.join(abspath.split('/')[3:])


def _get_ids_not_downloaded(id_list: List[str],
                            genomes_metadata: pd.DataFrame,
                            genomes_directory: Optional[str],
                            fna_only: Optional[bool] = False) -> List[str]:

    # if fna_only, we are checking for ids to gunzip
    # else, we are checking to see if we need to download the .gz file
    # results in the following truth table:
    # to_download | .fna exists | .fna.gz exists | fna_only
    #       T     |      F      |        F       |    F
    #       F     |      F      |        T       |    F
    #       F     |      T      |        T       |    F
    #       F     |      T      |        F       |    F
    #  err<-F     |      F      |        F       |    T
    #       T     |      F      |        T       |    T
    #       F     |      T      |        T       |    T
    #       F     |      T      |        F       |    T
    # throws value error if a requested id does not have a .fna or .fna.gz
    # and is asked whether it should gunzip the .fna.gz, likely indicates a
    # failed download earlier on

    if genomes_directory is None:
        genomes_directory = os.path.curdir()

    ids_to_download = []
    for id_ in id_list:
        expected_local_dir = os.path.join(genomes_directory,
                                          id_)
        fna_gz_name = get_fna_name(id_, genomes_metadata)
        fna_name = gz_stripper(fna_gz_name)

        try:
            existing_files = set(os.listdir(expected_local_dir))
            fna_present = fna_name in existing_files
            fna_gz_present = fna_gz_name in existing_files
            # the logic here is a little tricky, see note above
            if not fna_only:
                if not fna_present and not fna_gz_present:
                    ids_to_download.append(id_)
            else:
                if not fna_present and fna_gz_present:
                    ids_to_download.append(id_)
                elif not fna_present:
                    raise ValueError('Cannot gunzip when .gz file is not '
                                     'present, ID: {}'.format(id_))

        # if the directory does not exist, then create a directory for it
        # and add the id to the download list
        # TODO: more specific error catching?
        except FileNotFoundError:
            os.makedirs(expected_local_dir, exist_ok=True)
            ids_to_download.append(id_)

    return ids_to_download


def _assure_all_data(id_list: List[str],
                     genomes_metadata: pd.DataFrame,
                     genomes_directory: Optional[str]) -> List[str]:

    if genomes_directory is None:
        genomes_directory = os.path.curdir()

    possible_ids = set(genomes_metadata.index)
    for id_ in id_list:
        if id_ not in possible_ids:
            raise ValueError('Invalid assembly accession ID for this '
                             'channel: {}'.format(id_))

    ids_to_download = _get_ids_not_downloaded(id_list,
                                              genomes_metadata,
                                              genomes_directory)

    # download .fna.gz files we do not have (do not need to download if
    # .fna exists)

    _ncbi_ftp_downloader(ids_to_download, genomes_metadata, genomes_directory)

    # if .fna.gz files are not unzipped, unzip them
    ids_to_gunzip = _get_ids_not_downloaded(id_list, genomes_metadata,
                                            genomes_directory, fna_only=True)

    _file_gunzipper(ids_to_gunzip, genomes_metadata, genomes_directory)

    id_file_list = [(id_, gz_stripper(get_fna_name(id_, genomes_metadata)))
                    for id_ in id_list]

    return [os.path.join(genomes_directory, *pair) for pair in id_file_list]


def data_downloader(genome_ids: List[str],
                    genomes_directory: Optional[str] = None,
                    channel: Optional[str] = 'representative') -> List[str]:

    if channel == 'representative':
        genomes_metadata = pd.read_csv(representative_genomes_file,
                                       sep='\t', index_col=0)
    elif channel == 'complete':
        genomes_metadata = pd.read_csv(complete_genomes_file, sep='\t',
                                       index_col=0)
    else:
        raise ValueError("Invalid choice for `channel`. Options are "
                         "'representative' and 'complete'.")

    # make sure all genomes are downloaded (download if not)
    fasta_filenames = _assure_all_data(genome_ids,
                                       genomes_metadata,
                                       genomes_directory)

    return fasta_filenames

