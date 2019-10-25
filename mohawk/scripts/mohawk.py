#!/usr/bin/env python
import time
import os
import click
from mohawk.classify import classify as classify_dl
from mohawk.trainer import trainer
from mohawk.models import (SmallConvNet, ConvNet2, ConvNetAvg, ConvNetAvg2,
                           ConvNetAvg3, ConvNetAvg4, ConvNetAvg5, ConvNetAvg6)
import pandas as pd


@click.group()
def mohawk():
    pass


# TODO the sequence length should be determined by the model, and the error
#  is that the sequence data given does not have the right length
@mohawk.command()
@click.option('--model-path', type=click.Path(exists=True),
              help='A path to a trained model', required=True)
@click.option('--sequence-file', type=click.File('rb'),
              help='A sequence files', required=True)
@click.option('--output-file', type=click.File('wb'),
              help='The file to save the output', required=True)
@click.option('--length', type=int, default=150, required=False,
              help='The length of sequences to produce')
def classify(model_path, sequence_file, output_file, length):
    # TODO more options for fastq i/o
    format = 'fastq'
    format_kwargs = {'phred_offset': 33}
    results = classify_dl(model_path, length, sequence_file,
                          batch_size=640,
                          format_=format,
                          format_kwargs=format_kwargs)

    results.to_csv(output_file.name, sep='\t', index=False)


# TODO is there a way to do this programmatically?
model_names_to_obj = {'SmallConvNet': SmallConvNet,
                      'ConvNet2': ConvNet2,
                      'ConvNetAvg': ConvNetAvg,
                      'ConvNetAvg2': ConvNetAvg2,
                      'ConvNetAvg3': ConvNetAvg3,
                      'ConvNetAvg4': ConvNetAvg4,
                      'ConvNetAvg5': ConvNetAvg5,
                      'ConvNetAvg6': ConvNetAvg6
                      }


@mohawk.command()
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
def train(model_name, genome_ids, external_validation_ids, metadata, lr,
          epochs, summarize, log_dir, summary_interval, train_ratio, length,
          seed, concise_summary, gpu, batch_size, data_dir,
          additional_hyper_parameters):
    # TODO throw better error
    start_time = time.time()
    model = model_names_to_obj[model_name]

    id_list, distribution, classes, n_reads = id_file_loader(genome_ids)

    if external_validation_ids is not None:
        ext_ids, ext_dist, ext_classes, ext_reads = id_file_loader(
            external_validation_ids)
    else:
        ext_ids, ext_dist, ext_classes, ext_reads = None, None, None, None

    train_kwargs = {'gpu': gpu,
                    'learning_rate': lr,
                    'summary_interval': summary_interval,
                    'epochs': epochs,
                    'summarize': summarize,
                    'log_dir': log_dir,
                    }
    summary_kwargs = {'concise': concise_summary}

    if additional_hyper_parameters is not None:
        additional_hparams = parse_hparams_file(additional_hyper_parameters)
    else:
        additional_hparams = None

    trainer(model, n_reads, length, train_ratio, id_list=id_list,
            metadata=metadata, distribution=distribution, class_list=classes,
            batch_size=batch_size, data_directory=data_dir, random_seed=seed,
            external_validation_ids=ext_ids,
            n_external_validation_reads=ext_reads,
            external_validation_distribution=ext_dist,
            external_validation_classes=ext_classes,
            start_time=start_time, train_kwargs=train_kwargs,
            summary_kwargs=summary_kwargs,
            additional_hparams=additional_hparams)


def parse_hparams_file(fp):
    df = pd.read_csv(fp, sep='\t', index_col=0)
    dict_ = df.to_dict()['0']
    out_dict = dict()
    for key, val in dict_.items():
        # try to make the key numerical, but if not able to, leave as str
        try:
            out_dict[key] = float(val)
        except KeyError:
            out_dict[key] = val
    return out_dict


def id_file_loader(genome_ids):
    id_df = pd.read_csv(genome_ids, sep='\t')
    id_list = id_df['id']
    n_reads_by_genome = id_df['n_reads']
    n_reads = int(n_reads_by_genome.sum())
    distribution = n_reads_by_genome / n_reads
    if 'class' in id_df.columns:
        classes = id_df['class']
    else:
        classes = None

    return id_list, distribution, classes, n_reads


if __name__ == '__main__':
    mohawk()
