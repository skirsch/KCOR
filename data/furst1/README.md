# Furst test files

There are two furst directories under data/: furst1 and furst2

Each directory has a data.csv file.

We use the convert script to convert this krf, then process via the normal flow.

So from repo root, we can say and it will call the convert, CMR, and KCOR on the data using the config.yaml file for parameters.
- make KCOR DATASET=furst1
- make KCOR DATASET=furst2

## the data.csv files

There are around 1M lines and each line is the data for a person.

The first column is the date of death (in days from the beginning, 
i.e. number 45 means that the person died on day 45 from the starting date)
0 means the person did not die.

Approx 2% of the individuals die within the 700 modelled days.

The second column is the date of vaccination (in days from the beginning)

Approx 80% of the population get vaccinated around month 6 (gausian distribution of times).
0 means the person was not vaccinated.

I've adjusted the config.yaml file for the parameters for the data. Have a look.

So the key is in the convert file. 

Once it is converted to krf, the make file will call the krf converter to czech format and run it through the whole process.

I want to make sure convert.py is correct and that the make files will input and output to the right targets including the main Makefile and the local Makefile in this directory

I changed the paths to be relative paths in the yaml file (relative to the current directory)

so people got either no vaccine or one vaccine dose.
