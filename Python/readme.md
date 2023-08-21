One channel calculation

jpsi_1c.py code

Running:

>python jpsi_1c.py $option $dataset $nmc $lmax $leff ($file)

Last input is optional

$option

read      : Read fit parameters for the best fit and computes chi2 for each point and pull. If $file is provided, that's used for calculation, if not pcbff.txt is used.
fit       : Fit selected datasets. If $file is provided it is used as seed for the fit parameters. Output: pcbff.txt, fit parameters ordered according to chi2
bs        : Bootstrap calculation. If $file is provided, that's used as seed for the bootstrap, if not pcbff.txt is used. The bootstraped parameters are only those bootstraped in the paper. This can be changed modifying the variable 'fixated'. Output: pcbs.txt
plot      : Plot of the best fit. If $file is provided, that's used for computing observables, if not pcbff.txt is used
plotlog   : Plot of the best fit in log-y scale. If $file is provided, that's used for computing observables, if not pcbff.txt is used
plotbs    : Calculation and plotting of the observables from the bootstrap. If $file is provided, that's used for computing observables, if not pcbs.txt is used
plotlogbs : Calculation and plotting in log-y scale of the observables from the bootstrap. If $file is provided, that's used for computing observables, if not pcbs.txt is used. 
total     : Calculation and plotting of the total cross section. If $file is provided, that's used for computing observables, if not pcbff.txt is used
totalbs   : Calculation and plotting of the total cross section from the bootstrap. If $file is provided, that's used for computing observables, if not pcbs.txt is used

$dataset

If $option== fit, bs, plot, plotlog, plotbs, plotlogbs

gluex    : uses only GlueX dataset
007      : uses only Hall-C 007 dataset
combined : uses bth GlueX and Hall-C 007 datasets

$nmc

If $option==read irrelevant
If $option==fit           : provides the number of fits with randomly seeded parameters to find the best fit
If $option==bs            : provides the number of bootstrap fits to perform
If $option==plot, plotlog, total : Used to plot several best fits from Sfile (if $file is not provided pcbff.txt is used). all best fits from $nmc to $lmax are plotted. If $nmc =0 and $lmax=1 only absolute best fit is plotted
If $option==plotbs, plotlogbs : If $nmc==0 then files with the observables computed are read and plotted. If $nmc !=0 then observables are computed using the $file (if $file is not provided pcbs.txt is used). $dataset defines which observables are plotted against data. Output files if $nmc !=0: plot_xsec_gluex.txt (GlueX total cross section), plot_dsdt_gluex_?.txt (3 files with GlueX differential cross sections); plot_dsdt_007??.txt (files with Hall-C 007 differential cross sections) where ? and ?? are numbers. These same files are the ones read and used if $nmc==0

If $option==totalbs:  If $nmc==0 then files with the observables computed are read and plotted. If $nmc !=0 then observables are computed using the $file (if $file is not provided pcbs.txt is used). $dataset defines which observables are plotted against data. Output files if $nmc !=0:
plot_totalbs.txt (total cross section). This fils is the one read and used if $nmc==0

$lmax

If $option==read, bs, plotbs, plotlogbs, total, totalbs : irrelevant
If $option== fit                        : Number of partial waves used in the model for the fit
If $option==plot, plotlog               : all best fits from $nmc to $lmax are plotted. If $nmc =0 and $lmax=1 only absolute best fit is plotted


$leff

If $option==fit : Highest partial wave to include in effective range approximation. $leff=<$lmax
If $option==plot, plotlog, total : See $nmc usage
else irrelevant


Two channels calculation

jpsi_2c.py code

Running:

>python jpsi_2c.py $option $dataset $nmc $lmax $modelo ($file)

$option

read      : Read fit parameters for the best fit and computes chi2 for each point and pull. If $file is provided, that's used for calculation, if not pcbff.txt is used.
fit       : Fit selected datasets. If $file is provided it is used as seed for the fit parameters. Output: pcbff.txt, fit parameters ordered according to chi2
bs        : Bootstrap calculation. If $file is provided, that's used as seed for the bootstrap, if not pcbff.txt is used. The bootstraped parameters are only those bootstraped in the paper. This can be changed modifying the variable 'fixated'. Output: pcbs.txt
plot      : Plot of the best fit. If $file is provided, that's used for computing observables, if not pcbff.txt is used
plotlog   : Plot of the best fit in log-y scale. If $file is provided, that's used for computing observables, if not pcbff.txt is used
plotbs    : Calculation and plotting of the observables from the bootstrap. If $file is provided, that's used for computing observables, if not pcbs.txt is used
plotlogbs : Calculation and plotting in log-y scale of the observables from the bootstrap. If $file is provided, that's used for computing observables, if not pcbs.txt is used. 
total     : Calculation and plotting of the total cross section. If $file is provided, that's used for computing observables, if not pcbff.txt is used
totalbs   : Calculation and plotting of the total cross section from the bootstrap. If $file is provided, that's used for computing observables, if not pcbs.txt is used
polebff   : Calculation of poles and Riemann sheets for the best fits from $file (pcbff.txt if no $file is provided)
polecheck : Check if there are poles on the first Riemann sheet for the best fits from $file (pcbff.txt if no $file is provided)


$dataset

- $option== fit, bs, plot, plotlog, plotbs, plotlogbs

gluex    : uses only GlueX dataset
007      : uses only Hall-C 007 dataset
combined : uses bth GlueX and Hall-C 007 datasets


$nmc 

- $option==read          : irrelevant
- $option==fit           : provides the number of fits with randomly seeded parameters to find the best fit
- $option==bs            : provides the number of bootstrap fits to perform
- $option==plot, plotlog, total : Used to plot several best fits from Sfile (if $file is not provided pcbff.txt is used). all best fits from $nmc to $lmax are plotted. If $nmc =0 and $lmax=1 only absolute best fit is plotted
- $option==polecheck, polebff : Used to compute poles for several best fits from Sfile (if $file is not provided pcbff.txt is used). all best fits from $nmc to $lmax are plotted. If $nmc =0 and $lmax=1 only absolute best fit is plotted

$lmax

- $option==read, bs, plotbs, plotlogbs, total, totalbs : irrelevant
- $option==fit                         : Number of partial waves used in the model for the fit
- $option==polebff, polecheck, plot, plotlog, total : see $nmc

$modelo

init  : 
- $option==fit : allows to determine which parameters are free and which ones are fixed to zero
- $option==plotbs, plotlogbs, totalbs : computes the observables. Output files: plot_xsec_gluex.txt (GlueX total cross section), plot_dsdt_gluex_?.txt (3 files with GlueX differential cross sections); plot_dsdt_007??.txt (files with Hall-C 007 differential cross sections) where ? and ?? are numbers.
sfree :
- $option==fit : Fixes the higher partial waves using pcbff.txt file or $file if given and fits S wave
- rest of options equivalent to $modelo==init
scat2 : 
- $option==fit : selects the fit parameters used in the paper
- $option==plotbs, plotlogbs, totalbs : reads observables from files
a     : 
- $option==fit fixes the model to minimal open-charm contribution
- rest of options equivalent to $modelo==init
c     :
- $option==fit fixes the model to miximal open-charm contribution
- rest of options equivalent to $modelo==init



Three channels calculation

jpsi_3c.py code

Running:

>python jpsi_2c.py $option $dataset $nmc $lmax $modelo ($file)

$option

read      : Read fit parameters for the best fit and computes chi2 for each point and pull. If $file is provided, that's used for calculation, if not pcbff.txt is used.
fit       : Fit selected datasets. If $file is provided it is used as seed for the fit parameters. Output: pcbff.txt, fit parameters ordered according to chi2
bs        : Bootstrap calculation. If $file is provided, that's used as seed for the bootstrap, if not pcbff.txt is used. The bootstraped parameters are only those bootstraped in the paper. This can be changed modifying the variable 'fixated'. Output: pcbs.txt
plot      : Plot of the best fit. If $file is provided, that's used for computing observables, if not pcbff.txt is used
plotlog   : Plot of the best fit in log-y scale. If $file is provided, that's used for computing observables, if not pcbff.txt is used
plotbs    : Calculation and plotting of the observables from the bootstrap. If $file is provided, that's used for computing observables, if not pcbs.txt is used
plotlogbs : Calculation and plotting in log-y scale of the observables from the bootstrap. If $file is provided, that's used for computing observables, if not pcbs.txt is used. 
total     : Calculation and plotting of the total cross section. If $file is provided, that's used for computing observables, if not pcbff.txt is used
totalbs   : Calculation and plotting of the total cross section from the bootstrap. If $file is provided, that's used for computing observables, if not pcbs.txt is used
polebff   : Calculation of poles and Riemann sheets for the best fits from $file (pcbff.txt if no $file is provided)
polecheck : Check if there are poles on the first Riemann sheet for the best fits from $file (pcbff.txt if no $file is provided)
polebs    : compute all poles for the bootstrap from $file (pcbs.txt if no $file is provided)

$dataset

- $option== fit, bs, plot, plotlog, plotbs, plotlogbs

gluex    : uses only GlueX dataset
007      : uses only Hall-C 007 dataset
combined : uses bth GlueX and Hall-C 007 datasets


$nmc 

- $option==read          : irrelevant
- $option==fit           : provides the number of fits with randomly seeded parameters to find the best fit
- $option==bs            : provides the number of bootstrap fits to perform
- $option==plot, plotlog, total : Used to plot several best fits from Sfile (if $file is not provided pcbff.txt is used). all best fits from $nmc to $lmax are plotted. If $nmc =0 and $lmax=1 only absolute best fit is plotted
- $option==polecheck, polebff : Used to compute poles for several best fits from Sfile (if $file is not provided pcbff.txt is used). all best fits from $nmc to $lmax are plotted. If $nmc =0 and $lmax=1 only absolute best fit is plotted


$lmax

- $option==read, bs, plotbs, plotlogbs, total, totalbs : irrelevant
- $option==fit                         : Number of partial waves used in the model for the fit
- $option==polebff, polecheck, plot, plotlog, total : see $nmc

$modelo

init  : 
- $option==fit : allows to determine which parameters are free and which ones are fixed to zero
- $option==plotbs, plotlogbs, totalbs : computes the observables. Output files: plot_xsec_gluex.txt (GlueX total cross section), plot_dsdt_gluex_?.txt (3 files with GlueX differential cross sections); plot_dsdt_007??.txt (files with Hall-C 007 differential cross sections) where ? and ?? are numbers.

scat3 : 
- $option==fit : selects the fit parameters used in the paper
- $option==plotbs, plotlogbs, totalbs : reads observables from files


Input: Experimental data

sigma_gluex.txt
dsdt_gluex.txt
dsdt_jpsi007.csv


Output files:

pcbff.txt    : Parameters of the best fit
pcbs.txt     : Parameters of the bootstrap
bsdata.txt   : Pseudodata used for the bootstrap
polebs.txt   : Poles from the bootstrap
sigmatot.txt : Total cross section from the best fit
plot_totalbs.txt       : Total cross section from the bootstrap
plot_xsec_gluex.txt    : Total cross section from the bootstrap to compare with GlueX data
plot_dsdt_gluex_0.txt  : Differential cross section from the bootstrap to compare with GlueX data, lowest energy
plot_dsdt_gluex_1.txt  : Differential cross section from the bootstrap to compare with GlueX data, mid energy
plot_dsdt_gluex_2.txt  : Differential cross section from the bootstrap to compare with GlueX data, highest energy
plot_dsdt_007??.txt    : Differential cross sections from the bootstrap to compare with Hall C data


Figures:

plotgluex.pdf    : Observables from the best fit(s) compared to GlueX data
plot007.pdf      : Observables from the best fit(s) compared to Hall C data
plotbsgluex.pdf  : Observables from the bootstrap compared to GlueX data
plotbs007.pdf    : Observables from the bootstrap compared to Hall C data
polebs.pdf       : Poles from the bootstrap
sigmatotbs.pdf   : Total cross section from the bootstrap
sigmatot.pdf     : Total cross section from the best fit(s)

