# Sample scripts for RFA-OD-24-011

 ## 1. average_across_shap_files.py

### Calculates average contribution scores across multiple folds output from the [ChromBPNet pipeline](https://github.com/kundajelab/chrombpnet/wiki/Generate-contribution-score-bigwigs)

A typical ChromBPNet workflow for a single dataset involves: defining five folds with different combinations of 
training, test and validation chromosomes; training a model and calculating contribution scores (shap values) for 
each of these folds; averaging the shap values; and using these averaged shap values to identify transcription factor 
motifs with [TFModisco](https://github.com/jmschrei/tfmodisco-lite). In addition to averaging the shap values, this 
script optionally produces quality control matrix scatter plots for quickly comparing shap values between folds -- 
values should be correlated but not identical. 

## 2. make_jsons.py

### Produces json files with necessary metadata for uploading ChromBPNet models to the [ENCODE portal](https://www.encodeproject.org/)

ChromBPNet model [datasets](https://www.encodeproject.org/annotations/ENCSR432KJZ/) that are uploaded to the ENCODE portal 
include dozens of files; in addition to an `h5` and `tar` version of each model we've trained, we include regions 
bed files of train, test and validation data, numerous log files, bed files of background regions, a bigwig of the 
signal track, and others. This script automates the production of a json with metadata of file locations that is then 
ingested by another [script](https://github.com/kundajelab/encode_upload_scripts), which produces tar files that are 
then uploaded to the ENCODE portal.  
