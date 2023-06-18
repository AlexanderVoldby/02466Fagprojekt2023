#!/bin/sh
# embedded options to bsub - start with #BSUB
### -- set the job Name AND the job array --
#BSUB -J shiftNMF_urine[1-15]
### –- specify queue -- 
#BSUB -q hpc 
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- set walltime limit: hh:mm --
#BSUB -W 48:00 
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=30GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id %I is the job-array index --
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output.out
#BSUB -e ./error/Output_%J_%I.err 
# here follow the commands you want to execute 
# Program_name_and_options
source test-env/bin/activate

MODEL="shiftNMF"
DATASET="urine"

python hpc_lr_test.py $LSB_JOBINDEX $MODEL $DATASET >  ./output/out_$DATASET+$MODEL+$LSB_JOBINDEX.out



