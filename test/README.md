# test
This is for doing negative control tests.

there is a code directory with the program to generate the test file
then we call KCORv4.py to analyze it, setting the input and output file
there is an out directory with the output.

The initial negative control test is to compare unvaccinated with unvaccinated in each enrollment group.

We read the data/Czech/KCOR_CMR.xlsx file, and we create corresponding control sheets by using 10 year younger age groups for the doses.

so for each sheet, we just cover dose 0, 1, and 2:

dose 0= unvaxxed born in 1930

dose 1= unvaxxed born in 1940

dose 2= unvaxxed born in 1950

and we make everyone born in 1950 (it doesn't matter)

Then we do the same method for the vaccinated, but treat them as all born in 1940. But do the same thing where the 
dose 0 = vaxxed born in 1930
dose 1= vaxxed born in 1940
dose 2= vaxxed born in 1950

so basically, we are comparing cohorts of those in the same category, which should result in a very low signal for the unvaccinated, and a higher signal for the vaccinated, but nowhere near as high as unvax--> vax comparison.

So we have a code, data, out directory and a Makefile at the root. The makefile makes the test file as described above and puts it in data. It then calls KCOR to process it, having it put the output file in the out directory.
