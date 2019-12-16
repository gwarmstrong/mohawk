#!/usr/bin/env python
import os
import click
from mohawk.classify import classify as classify_dl
from mohawk.trainer import train_helper


@click.group()
def mohawk():
    pass


@mohawk.group()
def train():
    pass


@mohawk.group()
def characterize():
    pass


# TODO the sequence length should be determined by the model, and the error
#  is that the sequence data given does not have the right length
@characterize.command()
@click.option('--model-path', type=click.Path(exists=True),
              help='A path to a trained model', required=True)
@click.option('--sequence-file', type=click.File('rb'),
              help='A sequence files', required=True)
@click.option('--output-file', type=click.File('wb'),
              help='The file to save the output', required=True)
@click.option('--length', type=int, default=150, required=False,
              help='The length of sequences to produce')
def classify_seq_by_seq_pytorch(model_path, sequence_file, output_file,
                                length):
    # TODO more options for fastq i/o
    format = 'fastq'
    format_kwargs = {'phred_offset': 33}
    results = classify_dl(model_path, length, sequence_file,
                          batch_size=640,
                          format_=format,
                          format_kwargs=format_kwargs)

    results.to_csv(output_file.name, sep='\t', index=False)


# TODO is there a way to do this programmatically?
@train.command()
@click.option('--model-name', type=str,  # TODO check if it is in dict() ?
              help='The name of the model type to train on', required=True)
@click.option('--genome-ids', type=click.Path(exists=True),
              help='The file that includes genome ids to train on and the '
                   'number of reads to sample from each genome. Genomes '
                   'will be considered the same class, unless classes are '
                   'provided in an additional column',
              required=True)
@click.option('--external-validation-ids', default=None,
              type=click.Path(exists=True),
              help='A file containing genome ids that are used to evaluate '
                   'the model, but are not used for training. Same format as '
                   '`genome-ids`',
              required=False)  # TODO do check for whether it contains ids
# in `genome-ids`
@click.option('--metadata', type=click.Path(exists=True), default=None,
              help='File containing assembly accession id\'s and ftp paths '
                   'in the case that data needs to be downloaded')
@click.option('--lr', type=float, default=0.0001, required=False,
              help='Portion of simulation data to train on. (1-train_ratio) '
                   'will be used for assessing performance on the ')
@click.option('--epochs', type=int, default=100, required=False,
              help='Number of epochs to train models on')
@click.option('--summarize', type=bool, default=True, required=False,
              help='Whether to write out summaries')
@click.option('--log-dir', type=str, default='runs/', required=False,
              help='Directory to save model logs too')
@click.option('--summary-interval', type=int, default=5, required=False,
              help='Interval between epochs to summarize model on')
@click.option('--train-ratio', type=float, default=0.9, required=False,
              help='Portion of simulation data to train on. (1-train_ratio) '
                   'will be used for assessing performance on the ')
@click.option('--length', type=int, default=150, required=False,
              help='The length of sequences to produce')
@click.option('--seed', type=int, default=0, required=False,
              help='The random seed to use in data sampling')
@click.option('--concise-summary', type=bool, default=True, required=False,
              help='If True, only writes some select summary statistics.')
@click.option('--gpu', type=bool, default=False, required=False,
              help='Uses gpu (if available) when True')
@click.option('--batch-size', type=int, default=64, required=False,
              help='Size of batches to use for training')
@click.option('--data-dir', type=click.Path(exists=True),
              default=os.curdir, required=False,
              help="Directory to look for data or to store data when it is "
                   "downloaded")
@click.option('--additional-hyper-parameters', type=click.Path(exists=True),
              default=None, required=False,
              help="A tab separated file containing two columns, with the "
                   "name of hyper-parameter in the first column, and the "
                   "value in the second column, and no header.")
@click.option('--append-time', type=bool, default=True, required=False,
              help="Whether the time should be appended to the supplied "
                   "log_dir argument, which is useful for reducing log_dir "
                   "collision")
def seq_by_seq_pytorch(model_name, genome_ids, external_validation_ids,
                       metadata, lr, epochs, summarize, log_dir,
                       summary_interval, train_ratio, length, seed,
                       concise_summary, gpu, batch_size, data_dir,
                       additional_hyper_parameters, append_time):
    # TODO throw better error
    train_helper(model_name, genome_ids, external_validation_ids, metadata, lr,
                 epochs, summarize, log_dir, summary_interval, train_ratio,
                 length, seed, concise_summary, gpu, batch_size, data_dir,
                 additional_hyper_parameters, append_time)


if __name__ == '__main__':
    mohawk()
