# mohawk
let's take a bunch of letters and use them to make numbers, then see if we can use numbers to make letters

## TODO

[x] split taxonomy_slim into two files

    [x] one with just taxonomy/lineage for assembly_accession
    
    [x] one with just ftp link for assembly_accession
    
    [ ] Maybe it is actually better to have these together to simplify things
    
[ ] use  into_numbers framework to generate training data for a given trial

    [ ] use multinomial to sample from distribution of abundances
    
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

[ ] provide functionality for refreshing databases

[ ] have setup_genome_downloads.py run on install -> also make more elegant
