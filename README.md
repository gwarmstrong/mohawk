[![Build Status](https://travis-ci.org/gwarmstrong/mohawk.svg?branch=master)](https://travis-ci.org/gwarmstrong/mohawk)
[![Coverage Status](https://coveralls.io/repos/github/gwarmstrong/mohawk/badge.svg?branch=master)](https://coveralls.io/github/gwarmstrong/mohawk?branch=master)

*This code is still under active initial development, so major backwards-incompatible changes might occur*

# mohawk
Neural nets for taxonomic profiling of microbial communities from 
metagenomic shotgun sequencing data.


## Install
Currently this repo can be installed 
```bash
git clone https://github.com/gwarmstrong/mohawk.git
cd mohawk
conda env create -n mohawk -f resources/environment.yml
conda activate mohawk
pip install -e .
```

## Usage
### Command Line Interface (CLI)

An example of using the `mohawk` API to train a neural
network on some pre-specified genomes with designated
classes on a given number of reads, you can use a command
similar to the one shown below:

```bash
mohawk train --genome-ids example_input/min-red_0.1__n_4__group_0__genome-ids.txt \
    --model-name ConvNetAvg \
    --log-dir example_output \
    --data-dir example_input/genomes \
    --metadata example_input/wol_supplemental_metadata.tsv \
    --train-ratio 0.8 \
    --gpu True
```

An example of using the interface for using a pre-trained mohawk model 
to classify reads from a `fastq` file is included below:

```bash
mohawk classify --model /path/to/model \
    --sequence-file /path/to/input/fastq \
    --output-file /path/to/ouput/file \
    --length 150
```

## Metadata
An up-to-date assembly summary from RefSeq can be obtained from:
ftp://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt


