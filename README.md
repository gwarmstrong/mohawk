# mohawk
let's take a bunch of letters and use them to make numbers, then see if we can use numbers to make letters

## Environment Setup

```bash
conda env create -n mohawk -f resources/environment.yml
conda activate mohawk
```


## TODO

[x] split taxonomy_slim into two files

    [x] one with just taxonomy/lineage for assembly_accession
    
    [x] one with just ftp link for assembly_accession
    
    [ ] Maybe it is actually better to have these together to simplify things
    
[x] use  into_numbers framework to generate training data for a given trial

    [x] use multinomial to sample from distribution of abundances
    
        [ ] noisy (shouldn't align) reads?
        
    [ ] distributions of different sizes?
    
    [ ] paired end reads?

[x] write script for generating ftp_link/taxon files

[ ] start writing models

~~[ ] make data_downloader object oriented?~~

[ ] more/better unit testing

    [ ] trial generation should be transitioned into a unit test

[x] consider renaming data_generator to something else: like data_downloader
that indicates it is getting data from ncbi, not _generating_ it

[x] use hashing (`hashlib`) to determine if updates have been made to ncbi
    
    [ ] unit test building the ftp link database
    
[ ] include class level in data generation: (e.g., classes assigned by read, by experiment)

[x] ~~provide functionality for refreshing databases~~ can run setup_genome_downloads.py

[x] have setup_genome_downloads.py run on ~~install~~ when importing if resources do not exist -> also make more elegant

[ ] functionality for saving and reading args files -> and building models from these

[ ] inject level labels into names, i.e., 'g_Staphylococcus', 'f_...', etc

[ ] more thoroughly document requirements
