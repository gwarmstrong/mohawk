import os
import gzip
import shutil
import pandas as pd
from ftplib import FTP
from typing import List, Optional
from mohawk.utils import _ftp_path, get_zipped_fasta_name, gz_stripper, \
    default_metadata


def _ncbi_ftp_downloader(id_list: List[str],
                         genomes_metadata: pd.DataFrame,
                         genomes_directory: str) -> bool:
    """
    Opens an FTP and downloads the genomes of all ids in `id_list`
    """

    ftp = FTP('ftp.ncbi.nih.gov')
    ftp.login(user='anonymous', passwd='example@net.com')
    for id_ in id_list:
        abspath = _ftp_path(id_, genomes_metadata)
        ftp_dir = get_ftp_dir(abspath)
        ftp.cwd(ftp_dir)
        filename = get_zipped_fasta_name(id_, genomes_metadata)
        local_dir = os.path.join(genomes_directory, id_, filename)
        ftp.retrbinary("RETR " + filename, open(local_dir, 'wb').write)

    return True


def _gunzip(gz_file: str, gunzipped_file: str) -> bool:
    """
    unzips `gz_file` and saves the contents as `gunzipped_file`
    """
    # see 'https://stackoverflow.com/questions/48466421/python-how-to-'
    #     'decompress-a-gzip-file-to-an-uncompressed-file-on-disk'
    with gzip.open(gz_file, 'r') as f_in, \
            open(gunzipped_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    return True


# TODO flat directory structure
def _file_gunzipper(id_list: List[str],
                    genomes_metadata: pd.DataFrame,
                    genomes_directory: str) -> None:
    """Unzips the .gz file corresponding to each id in id_list"""

    for id_ in id_list:
        filename = get_zipped_fasta_name(id_, genomes_metadata)
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
    >>> abspath = 'ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/901/128/'\
                  '725/GCF_900128725.1_BCifornacula_v1.0'
    >>> get_ftp_dir(abspath)
    'genomes/all/GCF/900/128/725/GCF_900128725.1_BCifornacula_v1.0'

    """
    return '/' + '/'.join(abspath.split('/')[3:])


def _get_ids_not_downloaded(id_list: List[str],
                            genomes_metadata: pd.DataFrame,
                            genomes_directory: Optional[str],
                            to_unzip: Optional[bool] = False) -> List[str]:
    """Returns ID's from id_list that need to be downloaded/unzipped (do not
    already exist in `genomes_directory`)

    Parameters
    ----------
    id_list
        List of Assembly accession id's being requested from
    genomes_metadata
        Table containing metadata containing NCBI ID's and ftp links
    genomes_directory
        Directory to look for and save data into to
    to_unzip
        False if downloading, True if unzipping

    Returns
    -------

    List[str]
        Assembly accession id's that need to be downloaded from NCBI

    Raises
    ------

    ValueError
        If a supplied id is not contained in the requested metadata table

    """

    # if fasta_only, we are checking for ids to gunzip
    # else, we are checking to see if we need to download the .gz file
    # results in the following truth table:
    # to_download | .fna exists | .fna.gz exists | fasta_only
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

    # default save data in current directory
    if genomes_directory is None:
        genomes_directory = os.path.curdir()

    ids_to_download = []
    for id_ in id_list:
        # TODO make expected local_dir just genomes_directory...
        expected_local_dir = os.path.join(genomes_directory,
                                          id_)
        fasta_gz_name = get_zipped_fasta_name(id_, genomes_metadata)
        fasta_name = gz_stripper(fasta_gz_name)

        # will fail if the directory is not found
        try:
            existing_files = set(os.listdir(expected_local_dir))
            # TODO play with logic for readability
            fasta_present = fasta_name in existing_files
            fasta_gz_present = fasta_gz_name in existing_files
            # the logic here is a little tricky, see note above
            # TODO maybe flip clauses
            #  if need to download?
            if not to_unzip:
                if not fasta_present and not fasta_gz_present:
                    ids_to_download.append(id_)
            # if we need to unzip?
            else:
                if not fasta_present and fasta_gz_present:
                    ids_to_download.append(id_)
                elif not fasta_present:
                    raise ValueError('Cannot gunzip when .gz file is not '
                                     'present, ID: {}'.format(id_))

        # if the directory does not exist, then create a directory for it
        # and add the id to the download list
        # TODO: more specific error catching? or if statement for os.isdir?
        except FileNotFoundError:
            os.makedirs(expected_local_dir, exist_ok=True)
            ids_to_download.append(id_)

    return ids_to_download


def _ensure_all_data(id_list: List[str],
                     genomes_metadata: pd.DataFrame,
                     output_directory: str) -> List[str]:
    """

    Parameters
    ----------
    id_list
        A list of assembly accession id's to ensure from NCBI
    genomes_metadata
        Table containing metadata containing NCBI ID's and ftp links
    output_directory
        Directory to look for and save data into to

    Returns
    -------

    List[str]
        paths to the genome for each id requested

    Raises
    ------

    ValueError
        If a supplied id is not contained in the requested metadata table

    """

    # TODO flat directory structure
    ids_to_download = _get_ids_not_downloaded(id_list,
                                              genomes_metadata,
                                              output_directory)

    # download .fna.gz files we do not have (do not need to download if
    # .fna exists)
    # TODO flat directory structure
    _ncbi_ftp_downloader(ids_to_download, genomes_metadata, output_directory)

    # if .fna.gz files are not unzipped, unzip them
    ids_to_gunzip = _get_ids_not_downloaded(id_list, genomes_metadata,
                                            output_directory, to_unzip=True)

    # TODO flat directory
    _file_gunzipper(ids_to_gunzip, genomes_metadata, output_directory)

    # TODO flat directory structure for `get_zipped_fasta_name`
    id_file_list = [(id_, gz_stripper(get_zipped_fasta_name(id_,
                                                            genomes_metadata)))
                    for id_ in id_list]

    return [os.path.join(output_directory, *pair) for pair in id_file_list]


def data_downloader(genome_ids: List[str],
                    output_directory: Optional[str] = None,
                    metadata: Optional[str] = None) -> List[str]:
    """

    Parameters
    ----------
    genome_ids
        A list of assembly accession id's
    output_directory
        Directory to look for and save data into to
    metadata
        A file containing metadata for the genomes to be downloaded

    Returns
    -------

    List[str]
        The filepaths to the fasta files for each id requested.

    Raises
    ------

    ValueError
        If an invalid channel is selected

    """
    metadata_cols = ['ftp_path', '# assembly_accession']
    if metadata is None:
        genomes_metadata = pd.read_csv(default_metadata(),
                                       sep='\t', index_col=False)
    elif os.path.exists(metadata):
        genomes_metadata = pd.read_csv(metadata, sep='\t',
                                       index_col=False)
        if not all(genomes_metadata.columns.contains(val_) for val_ in
                   metadata_cols):
            raise ValueError("metadata must at least contain columns "
                             "for all of the following: {}"
                             .format(metadata_cols))
    else:
        raise ValueError("Argument `metadata` must be a valid filepath or "
                         "default `None`")

    if output_directory is None:
        output_directory = os.path.curdir

    print(genomes_metadata.head())
    genomes_metadata.set_index('# assembly_accession', inplace=True)
    possible_ids = set(genomes_metadata.index)
    for id_ in genome_ids:
        if id_ not in possible_ids:
            raise ValueError('Assembly accession ID \'{}\' is not in metadata'
                             .format(id_))

    # make sure all genomes are downloaded (download if not)
    fasta_filenames = _ensure_all_data(genome_ids,
                                       genomes_metadata,
                                       output_directory)

    return fasta_filenames
