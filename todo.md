## TODO

- [x] split taxonomy_slim into two files

    - [x] one with just taxonomy/lineage for assembly_accession
    
    - [x] one with just ftp link for assembly_accession
    
    - [ ] Maybe it is actually better to have these together to simplify things
    
- [x] use  into_numbers framework to generate training data for a given trial

    - [x] use multinomial to sample from distribution of abundances
    
        - [ ] noisy (shouldn't align) reads?
        
    - [ ] distributions of different sizes?
    
    - [ ] paired end reads?

- [x] write script for generating ftp_link/taxon files

- [x] start writing models

- [x] ~~make data_downloader object oriented?~~ fine without

- [ ] more/better unit testing

    - [x] ~~trial generation should be transitioned into a unit test~~ made it into a script instead
   
    Files:  
    First pass 
    
    - [x] data_downloader.py
    
    - [x] dataset.py
    
    - [ ] models.py
    
    - [ ] setup_genome_downloads.py (needs major [re]structuring)
    
    - [ ] simulation.py
    
    - [ ] trainer.py
    
    - [ ] utils.py
    
- [ ] more docs

    Files:  
    
    - [x] data_downloader.py
    
    - [x] dataset.py
    
    - [ ] models.py
    
    - [ ] setup_genome_downloads.py (needs major [re]structuring)
    
    - [ ] simulation.py
    
    - [ ] trainer.py
    
    - [ ] utils.py
    
- [ ] [re]structuring setup_genome_downloads.py
    
- [ ] transition trial.py into `click`-based CLI

- [x] at least move both (all 4) resource loaders into their own file

- [x] consider renaming data_generator to something else: like data_downloader
that indicates it is getting data from ncbi, not _generating_ it

- [x] use hashing (`hashlib`) to determine if updates have been made to ncbi
    
    - [ ] unit test building the ftp link database
    
- [ ] include class level in data generation: (e.g., classes assigned by read, by experiment)

- [x] ~~provide functionality for refreshing databases~~ can run setup_genome_downloads.py

- [x] have setup_genome_downloads.py run on ~~install~~ when importing if resources do not exist -> also make more elegant

- [ ] functionality for saving and reading args files -> and building models from these

- [ ] inject level labels into names, i.e., 'g_Staphylococcus', 'f_...', etc

- [x] more thoroughly document requirements

- [ ] save output model(s)?

- [x] Evaluate with F1 Score

- [x] Visualize computational graph

- [x] show softmax histogram
    
    - [x] split by right and wrong?
    
- [x] figure out how I should pass in validation clades

- [x] fix TypeError at end of iteration -> caused by summarize call outside in `trainer.py`

- [x] ~~check seeding, seems slightly off, could be related to conv1d?~~ looks fine

- [ ] custom tensorboard plot for % reads classified and portion classified for each class

- [ ] incorporating data from CAMI-sim

- [ ] make external validation more elegant

- [x] something is not scaling well because running out of memory on GPU even with dataloader

    - [x] Priority: HIGH
    
    - [x] ~~could be related to summaries?~~ -> definitely related to summaries, but I am not sure why. 
         For now I have just simplified the summary output
    
        - [x] put optional summary parameter

- [ ] classify all levels at one?

- [ ] write function to harvest nvidia-smi output throughout execution

- [x] Is there a way to do the evaluation with fewer forward passes?

- [ ] add summary scalars, such as learning rate, etc.

- [x] add confusion matrices ( https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard )

    - part of external validation not being as good could be explained by class imbalance
    
- [ ] try fully connected neural net on flattened channels 

    * i.e. 0, 1   ---\
           1, 0   ---/   0, 1, 1, 0 
           
- [ ] How to deal with reads of varying length?
    
    - [ ] Probably trimming ends/adding padding
    
- [x] Adjust for class imbalance
    
    - [x] can probably remove noise from multinomial for number of reads (better control)
    
    - [x] loss function adjustment by class ? 
    
- [ ] plot taxonomy on confusion matrix ?

- [x] ~~remove unzipping part, since `skbio.io.read` can handle zipped files~~ had some issues so abandoned

- [x] Add more unit tests

- [x] Project summary

- [x] remedy memory ballooning 

- [ ] draft benchmarks

- [ ] train on more genomes

- [ ] potential: classify at multiple levels (e.g., 50% genus, 20% spec1, 20% spec2, 10% unknown)