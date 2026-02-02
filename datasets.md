The actual input data for this project consists of raw LC-MS/MS mass spectrum files (.raw) and their derived chromatogram files (.mzML / .mgf); the remaining .txt / .fasta files are used for grouping information, validation, or comparison with the results of the original paper, but are not used as direct inputs for the self-supervised model.

Core Input: 

1️.raw files (raw data source) 

09062023_Mehta_GR10000524_DDRC_Sample1_480_cirrhotic.raw 

09062023_Mehta_GR10000524_DDRC_Sample2_491_cirrhotic.raw 

09062023_Mehta_GR10000524_DDRC_Sample3_554_cirrhotic.raw 

09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic.raw 

09062023_Mehta_GR10000524_DDRC_Sample5_654_cirrhotic.raw


09062023_Mehta_GR10000524_DDRC_Sample6_0121_HCC.raw 

09062023_Mehta_GR10000524_DDRC_Sample7_0187_HCC.raw 

09062023_Mehta_GR10000524_DDRC_Sample8_0203_HCC.raw 

09062023_Mehta_GR10000524_DDRC_Sample9_0206_HCC.raw 

09062023_Mehta_GR10000524_DDRC_Sample10_0543_HCC.raw 


Biological implications: 

LC-MS/MS data from serum sources

Two groups: 

cirrhotic (control)

HCC (hepatocellular carcinoma + cirrhotic background)


2️. mzML (intermediate format).

raw → .mzML 

Role: 

Open format 

Retains complete MS1 / MS2 information 


Role in project: 

Intermediate conversion format 

Optional input (if you prefer to read directly from mzML)


3️. mgf (Direct input for your self-supervised model) 
BEGIN IONS
PEMASS=...
CHARGE=...
RTINSECONDS=...
m/z intensity
m/z intensity
...
END IONS 

In this project, the input to the self-supervised model is MS/MS spectra in MGF format, with each BEGIN IONS … END IONS block considered as a training sample.
Data format seen by the model:
A spectrum = a set of (m/z, intensity) pairs

Can be regarded as:
Sequence
Set
Sparse
signal
