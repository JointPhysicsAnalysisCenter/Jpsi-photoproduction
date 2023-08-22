# Python codes

There are three versions of the code depending on how many channels are included in the model:
- One channel: <code>jpsi_1c.py</code>
- Two channels: <code>jpsi_2c.py</code>
- Three channels: <code>jpsi_3c.py</code>

All three codes can be found in the <code> codes </code> folder

The codes only need as input a set of options and the files with the Hall-C and GlueX data. Datafiles can be found in the  <code> data </code> folder and have to be in the same folder as the python code: sigma_gluex.txt, dsdt_gluex.txt, and dsdt_jpsi007.csv

The files with the results used in the paper can be found in the folder <code>results</code>

All questions and comments on the Python codes and files can be sent to Cesar Fernandez-Ramirez: cefera@ccia.uned.es

## Outputs

Regardless of the code the outputs are

Output files:

- pcbff.txt    : Parameters of the best fit
- pcbs.txt     : Parameters of the bootstrap
- bsdata.txt   : Pseudodata used for the bootstrap
- polebs.txt   : Poles from the bootstrap (only if the <code>polebs</code> option is available)
- sigmatot.txt : Total cross section from the best fit (only if the <code>total</code> option is available)
- plot_totalbs.txt       : Total cross section from the bootstrap (only if the <code>totalbs</code> option is available)
- plot_xsec_gluex.txt    : Total cross section from the bootstrap to compare with GlueX data
- plot_dsdt_gluex_0.txt  : Differential cross section from the bootstrap to compare with GlueX data, lowest energy
- plot_dsdt_gluex_1.txt  : Differential cross section from the bootstrap to compare with GlueX data, mid energy
- plot_dsdt_gluex_2.txt  : Differential cross section from the bootstrap to compare with GlueX data, highest energy
- plot_dsdt_007??.txt    : Differential cross sections from the bootstrap to compare with Hall C data


Output figures:

- plotgluex.pdf    : Observables from the best fit(s) compared to GlueX data
- plot007.pdf      : Observables from the best fit(s) compared to Hall C data
- plotbsgluex.pdf  : Observables from the bootstrap compared to GlueX data
- plotbs007.pdf    : Observables from the bootstrap compared to Hall C data
- polebs.pdf       : Poles from the bootstrap
- sigmatotbs.pdf   : Total cross section from the bootstrap (only if the <code>totalbs</code> option is available)
- sigmatot.pdf     : Total cross section from the best fit(s)  (only if the <code>total</code> option is available)

## Libraries

- <code>copy</code>
- <code>sys</code>
- <code>numpy</code>
- <code>matplotlib.pyplot</code>
- <code>from argparse import Namespace</code>
- <code>from iminuit import Minuit</code>


## One channel code

Run: <code> >python jpsi_1c.py $option $dataset $nmc $lmax $leff ($file) </code>

Last input is optional

<code>$option</code>

- <code>read</code>      : Read fit parameters for the best fit and computes $\chi^2$ for each point and pull. If <code>$file</code> is provided, that's used for calculation, if not pcbff.txt is used.
- <code>fit</code>       : Fit selected datasets. If <code>$file</code> is provided it is used as seed for the fit parameters. Output: pcbff.txt, fit parameters ordered according to $\chi^2$
- <code>bs </code>       : Bootstrap calculation. If <code>$file</code> is provided, that's used as seed for the bootstrap, if not pcbff.txt is used. The bootstraped parameters are only those bootstraped in the paper. This can be changed modifying the variable <code>fixated</code>. Output: pcbs.txt
- <code>plot </code>     : Plot of the best fit. If <code>$file</code> is provided, that's used for computing observables, if not pcbff.txt is used
- <code>plotlog </code>  : Plot of the best fit in log-y scale. If <code>$file</code> is provided, that's used for computing observables, if not pcbff.txt is used
-<code> plotbs</code>    : Calculation and plotting of the observables from the bootstrap. If <code>$file</code> is provided, that's used for computing observables, if not pcbs.txt is used
- <code>plotlogbs</code> : Calculation and plotting in log-y scale of the observables from the bootstrap. If <code>$file</code> is provided, that's used for computing observables, if not pcbs.txt is used. 
- <code>total</code>     : Calculation and plotting of the total cross section. If <code>$file</code> is provided, that's used for computing observables, if not pcbff.txt is used
- <code>totalbs</code>   : Calculation and plotting of the total cross section from the bootstrap. If <code>$file</code> is provided, that's used for computing observables, if not pcbs.txt is used

<code>$dataset</code>

If <code>$option in [fit, bs, plot, plotlog, plotbs, plotlogbs]</code>

- <code>gluex</code>    : uses only GlueX dataset
- <code>007</code>      : uses only Hall-C 007 dataset
- <code>combined</code> : uses both GlueX and Hall-C 007 datasets

</ode>$nmc</code>

- <code>$option==read</code>: irrelevant
- <code>$option==fit</code> : provides the number of fits with randomly seeded parameters to find the best fit
- <code>$option==bs </code> : provides the number of bootstrap fits to perform
- <code>$option in [plot, plotlog, total] </code>: Used to plot several best fits from <code>Sfile</code> (if <code>$file</code> is not provided pcbff.txt is used). All best fits from </ode>$nmc</code> to <code>$lmax</code> are plotted. If <code>$nmc ==0</code> and <code>$lmax=1</code> only absolute best fit is plotted
- <code>$option in [plotbs, plotlogbs]</code> : If <code>$nmc==0</code> then files with the observables computed are read and plotted. If $nmc !=0 then observables are computed using the <code>$file</code> (if <code>$file</code> is not provided pcbs.txt is used). <code>$dataset</code> defines which observables are plotted against data. Output files if <code>$nmc !=0</code>: plot_xsec_gluex.txt (GlueX total cross section), plot_dsdt_gluex_?.txt (3 files with GlueX differential cross sections); plot_dsdt_007??.txt (files with Hall-C 007 differential cross sections) where ? and ?? are numbers. These same files are the ones read and used if <code>$nmc==0</code>
-  <code>$option==totalbs</code>:  If <code>$nmc==0</code> then files with the observables computed are read and plotted. If <code>$nmc !=0</code> then observables are computed using the <code>$file</code> (if <code>$file</code> is not provided pcbs.txt is used). <code>$dataset</code> defines which observables are plotted against data. Output files if <code>$nmc !=0</code>. plot_totalbs.txt (total cross section). This fils is the one read and used if <code>$nmc==0</code>

<code>$lmax</code>

- <code>$option in [read, bs, plotbs, plotlogbs, total, totalbs]</code> : irrelevant
- <code>$option== fit</code>: Number of partial waves used in the model for the fit
- <code>$option in [plot, plotlog]</code>: all best fits from $nmc to $lmax are plotted. If <code>$nmc==0</code> and <code>$lmax==1</code> only absolute best fit is plotted


<code>$leff</code>

- <code>$option==fit</code> : Highest partial wave to include in effective range approximation. <code>$leff=<$lmax</code>
- <code>$option in [plot, plotlog, total]</code> : See <code>$nmc</code> usage
else irrelevant


## Two channels code

Run: <code> >python jpsi_2c.py $option $dataset $nmc $lmax $modelo ($file)</code>

<code>$option</code>

- <code>read</code>: Read fit parameters for the best fit and computes $\chi^2$ for each point and pull. If <code>$file</code> is provided, that's used for calculation, if not pcbff.txt is used.
- <code>fit</code>: Fit selected datasets. If <code>$file</code> is provided it is used as seed for the fit parameters. Output: pcbff.txt, fit parameters ordered according to $\chi^2$
- <code>bs</code>: Bootstrap calculation. If <code>$file</code> is provided, that's used as seed for the bootstrap, if not pcbff.txt is used. The bootstraped parameters are only those bootstraped in the paper. This can be changed modifying the variable <code>fixated</code>. Output: pcbs.txt
- <code>plot</code>: Plot of the best fit. If <code>$file</code> is provided, that's used for computing observables, if not pcbff.txt is used
- <code>plotlog</code>: Plot of the best fit in log-y scale. If <code>$file</code> is provided, that's used for computing observables, if not pcbff.txt is used
- <code>plotbs</code>: Calculation and plotting of the observables from the bootstrap. If <code>$file</code> is provided, that's used for computing observables, if not pcbs.txt is used
- <code>plotlogbs</code>: Calculation and plotting in log-y scale of the observables from the bootstrap. If <code>$file</code> is provided, that's used for computing observables, if not pcbs.txt is used. 
- <code>total</code>: Calculation and plotting of the total cross section. If <code>$file</code> is provided, that's used for computing observables, if not pcbff.txt is used
- <code>totalbs</code>: Calculation and plotting of the total cross section from the bootstrap. If <code>$file</code> is provided, that's used for computing observables, if not pcbs.txt is used
- <code>polebff</code>: Calculation of poles and Riemann sheets for the best fits from <code>$file</code> (pcbff.txt if no <code>$file</code> is provided)
- <code>polecheck</code>: Check if there are poles on the first Riemann sheet for the best fits from <code>$file</code> (pcbff.txt if no <code>$file</code> is provided)


<code>$dataset</code>

- <code>$option in  [fit, bs, plot, plotlog, plotbs, plotlogbs]</code>

- <code>gluex</code>    : uses only GlueX dataset
- <code>007</code>      : uses only Hall-C 007 dataset
- <code>combined</code> : uses both GlueX and Hall-C 007 datasets


<code>$nmc </code>

- <code>$option==read</code>: irrelevant
- <code>$option==fit</code>: provides the number of fits with randomly seeded parameters to find the best fit
- <code>$option==bs</code>: provides the number of bootstrap fits to perform
- <code>$option in [plot, plotlog, total]</code> : Used to plot several best fits from <code>Sfile</code> (if <code>$file</code> is not provided pcbff.txt is used). all best fits from <code>$nmc</code> to <code>$lmax</code> are plotted. If <code>$nmc==0</code> and <code>$lmax==1</code> only absolute best fit is plotted
- <code>$option in [polecheck, polebff]</code> : Used to compute poles for several best fits from Sfile (if <code>$file</code> is not provided pcbff.txt is used). all best fits from <code>$nmc</code> to <code>$lmax</code> are plotted. If <code>$nmc==0</code> and <code>$lmax==1</code> only absolute best fit is plotted

<code>$lmax</code>

- <code>$option in [read, bs, plotbs, plotlogbs, total, totalbs]</code>: irrelevant
- <code>$option==fit</code>: Number of partial waves used in the model for the fit
- <code>$option in [polebff, polecheck, plot, plotlog, total]</code>: see <code>$nmc</code>

<code>$modelo</code>

- <code>init</code>: <code>$option==fit</code>: allows to determine which parameters are free and which ones are fixed to zero; <code>$option in [plotbs, plotlogbs, totalbs]</code>: computes the observables. Output files: plot_xsec_gluex.txt (GlueX total cross section), plot_dsdt_gluex_?.txt (3 files with GlueX differential cross sections); plot_dsdt_007??.txt (files with Hall-C 007 differential cross sections) where ? and ?? are numbers.
- <code>sfree</code>: <code>$option==fit</code>: Fixes the higher partial waves using pcbff.txt file or <code>$file</code> if given and fits S wave; rest of options equivalent to <code>$modelo==init</code>
- <code>scat2</code>: <code>$option==fit</code>: selects the fit parameters used in the paper; <code>$option in [plotbs, plotlogbs, totalbs]</code>: reads observables from files
- <code>a</code>: <code>$option==fit</code> fixes the model to minimal open-charm contribution; rest of options equivalent to <code>$modelo==init</code>
- <code>c</code>: <code>$option==fit</code> fixes the model to miximal open-charm contribution; rest of options equivalent to <code>$modelo==init</code>



## Three channels code

Run: <code> >python jpsi_2c.py $option $dataset $nmc $lmax $modelo ($file)</code>

<code>$option</code>

- <code> read</code>       : Read fit parameters for the best fit and computes $\chi^2$ for each point and pull. If <code>$file</code> is provided, that's used for calculation, if not pcbff.txt is used.
- <code> fit</code>        : Fit selected datasets. If <code>$file</code> is provided it is used as seed for the fit parameters. Output: pcbff.txt, fit parameters ordered according to $\chi^2$
- <code> bs</code>         : Bootstrap calculation. If <code>$file</code> is provided, that's used as seed for the bootstrap, if not pcbff.txt is used. The bootstraped parameters are only those bootstraped in the paper. This can be changed modifying the variable <code>fixated</code>. Output: pcbs.txt
- <code> plot</code>       : Plot of the best fit. If <code>$file</code> is provided, that's used for computing observables, if not pcbff.txt is used
- <code> plotlog</code>    : Plot of the best fit in log-y scale. If <code>$file</code> is provided, that's used for computing observables, if not pcbff.txt is used
- <code> plotbs</code>     : Calculation and plotting of the observables from the bootstrap. If <code>$file</code> is provided, that's used for computing observables, if not pcbs.txt is used
- <code> plotlogbs</code>  : Calculation and plotting in log-y scale of the observables from the bootstrap. If <code>$file</code> is provided, that's used for computing observables, if not pcbs.txt is used. 
- <code> total</code>      : Calculation and plotting of the total cross section. If <code>$file</code> is provided, that's used for computing observables, if not pcbff.txt is used
- <code> totalbs</code>    : Calculation and plotting of the total cross section from the bootstrap. If <code>$file</code> is provided, that's used for computing observables, if not pcbs.txt is used
- <code> polebff</code>    : Calculation of poles and Riemann sheets for the best fits from <code>$file</code> (pcbff.txt if no <code>$file</code> is provided)
- <code> polecheck</code>  : Check if there are poles on the first Riemann sheet for the best fits from <code>$file</code> (pcbff.txt if no <code>$file</code> is provided)
- <code> polebs</code>     : compute all poles for the bootstrap from <code>$file</code> (pcbs.txt if no <code>$file</code> is provided)

<code>$dataset</code>

<code>$option in [fit, bs, plot, plotlog, plotbs, plotlogbs]</code>

- <code>gluex</code>    : uses only GlueX dataset
- <code>007</code>      : uses only Hall-C 007 dataset
- <code>combined</code> : uses both GlueX and Hall-C 007 datasets


<code>$nmc </code>

- <code>$option==read</code> : irrelevant
- <code>$option==fit</code>  : provides the number of fits with randomly seeded parameters to find the best fit
- <code>$option==bs</code> : provides the number of bootstrap fits to perform
- <code>$option in [plot, plotlog, total]</code>  : Used to plot several best fits from <code>$file</code>  (if <code> $file</code>  is not provided pcbff.txt is used). all best fits from <code> $nmc</code>  to <code> $lmax</code>  are plotted. If <code>$nmc==0</code>  and <code> $lmax=1</code>  only absolute best fit is plotted
- <code>$option in [polecheck, polebff]</code>  : Used to compute poles for several best fits from <code>$file</code> (if <code> $file</code>  is not provided pcbff.txt is used). all best fits from <code> $nmc</code> to <code>$lmax</code>  are plotted. If <code> $nmc==0</code>  and <code> $lmax==1</code>  only absolute best fit is plotted


<code>$lmax</code>

- <code> $option==fit</code> : Number of partial waves used in the model for the fit
- <code> $option in [polebff , polecheck, plot, plotlog, total]</code>  : see <code> $nmc</code> 
- rest: irrelevant

<code>$modelo</code>

- <code>init</code> : <code> $option==fit</code> : allows to determine which parameters are free and which ones are fixed to zero; <code> $option in [plotbs, plotlogbs, totalbs]</code>: computes the observables. Output files: plot_xsec_gluex.txt (GlueX total cross section), plot_dsdt_gluex_?.txt (3 files with GlueX differential cross sections); plot_dsdt_007??.txt (files with Hall-C 007 differential cross sections) where ? and ?? are numbers.

- <code>scat3</code> : <code>$option==fit</code> : selects the fit parameters used in the paper; <code>$option in [plotbs, plotlogbs, totalbs]</code> : reads observables from files
