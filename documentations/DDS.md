### Objective

For each MS/MS spectrum, calculate an embedding vector of a fixed dimension.

### Example

- Input: one MS/MS spectrum (m/z–intensity list)
- Intermediate output: embedding ∈ R^d (e.g., d = 128)
- Final output: predicted disease label ∈ {cirrhosis, HCC}
---

### add module 0 and extra filters.

## Calculation Steps


### Module 1 — Read the MGF file

### notify the users if files are bad log the issues
Parse MGF-formatted files and convert each MS/MS spectrum into a standardized in-memory spectrum object containing metadata and peak lists.

**Input**

- File: `*.mgf`
- Format: plain text file containing multiple blocks:
  - BEGIN IONS
  - m/z intensity
  - m/z intensity
  - ...
  - END IONS
 
**Parsing procedure**
- Read the MGF file line by line
- Detect spectrum boundaries using `BEGIN IONS` and `END IONS`
- Parse peak lines into numeric `(m/z, intensity)` pairs
- Store metadata and peaks in a standardized spectrum object

**Output**

- A list of spectrum objects (one per BEGIN IONS ... END IONS block)

**Output format**

- spectra: List[Spectrum]
- Spectrum = {
  - "meta": Dict[str, Any],      # e.g., {"PEPMASS": 512.34, "CHARGE": 2, "RTINSECONDS": 1234.5}
  - peaks": List[Tuple[float, float]]  # [(mz1, inten1), (mz2, inten2), ...]
- }  
### Module 2 — Extract the (m/z, intensity) pairs from each spectrum

Filter raw peak lists from spectrum objects and produce cleaned numeric representations suitable for downstream preprocessing and binning.

**Input**
- One Spectrum object from Step 1
- Spectrum["peaks"] = List[(mz, intensity)]

**Filtering**
- remove zero or negative intensity
- restrict m/z range
- keep top-N peaks

**Output**
- Numeric peak list (possibly cleaned or filtered)

**Output format**
- peaks: Array[N, 2]
 - column 0 = m/z, column 1 = intensity

### Module 3 — Discretize / tokenize the spectrum (m/z binning)

Convert variable-length peak lists into fixed-length vector representations by discretizing the m/z axis into bins and aggregating intensities.

**Input**

- peaks: Array[N, 2]
- Binning parameters:
  - mz_min, mz_max (e.g., 100–2000)
  - bin_width (e.g., 1.0 or 0.1 Da)
  
**Binning procedure**
- Map variable-length peak lists to fixed-length vectors by m/z binning and intensity aggregation
  
**Output**
- A fixed-length intensity vector x of length B (number of bins)
  
**Output format**
- x: Vector[B]
- aggregated (sum or max) and often normalized intensity per bin

### Module 4 — masked self-supervised learning

Masking is applied only during self-supervised pretraining. During downstream inference and classification, no masking is applied. This module defines the masking strategy and prepares masked inputs and reconstruction targets for self-supervised representation learning.

**Input**

- Binned vector x: Vector[B]
- mask_ratio (e.g., 0.15)

**Mask strategies**
- random bins, where individual bins are independently selected for masking
- structured masking (contiguous ranges), where adjacent m/z bins are masked together to simulate missing spectral regions
  
**Output**
- x_masked: Vector[B] (masked input)
- mask: Vector[B] (binary indicator: 1 = masked bin)
- targets: Vector[B] (ground truth values for masked bins only)


### Module 5 — Train the model to predict masked peaks (most important module)
This training step is fully self-supervised and does not use disease labels (HCC vs cirrhosis). The goal of this module is to train an encoder to learn general spectral representations by reconstructing masked spectral information.

**Model input**
- x_masked (or masked token sequence): Tensor[batch, B]
- mask (to compute loss only on masked locations): Tensor[batch, B]
- targets: Tensor[batch, B]

**Training procedure**
- Feed masked spectral representations into the encoder–decoder model
- Predict the original intensities at masked positions
- Compute reconstruction loss only on masked bins
- Update model parameters to learn general spectral representations without using disease labels
  
**Model output (training)**
- y_pred: reconstructed intensities (or token logits) over the full vector/sequence
- loss: computed on masked positions only

**output format**
- y_pred: Tensor[batch, B]
- loss:   float


### Module 6 — Extract spectrum-level embeddings
After self-supervised pretraining, the pretrained encoder is used to extract spectrum-level embeddings. This module applies the frozen or fine-tuned encoder to unmasked spectra to generate fixed-dimensional embedding vectors for downstream analysis.

**Input**
- unmasked or lightly masked spectrum representation (x or tokenized form)
  - x: Vector[B] or (token_ids, token_values, attention_mask)

**Embedding extraction procedure**
- Apply the pretrained encoder to unmasked spectra
- Extract fixed-dimensional spectrum-level embeddings

**Output**
- A spectrum embedding vector:
  - embedding ∈ R^d (e.g., d = 128)

### Downstream Tasks

### Module 7 — Aggregate spectrum-level embeddings to sample-level

Aggregate multiple spectrum-level embeddings derived from the same raw file into a single sample-level embedding representing the entire biological sample.

**Input**
- A set of embeddings from one raw file: (Each embedding corresponds to one MS/MS spectrum from the same sample.)
  - E = {e1, e2, ..., en},  ei ∈ R^d

**Aggregation methods**
- Mean pooling (used in this project)
- Median pooling
- Attention-based pooling (optional)

**Output**
- A single sample-level embedding:
  - z ∈ R^d

  
**Output format**
- sample_embedding: Vector[d]
  
**Note**

Due to limited computational resources:

1 cirrhotic raw file -> 1 sample embedding

1 HCC raw file -> 1 sample embedding

This setup is sufficient for validating the proposed pipeline.

### Module 8 — Train a classifier on sample-level embeddings
This step is a downstream supervised task that uses the learned sample-level embeddings as fixed representations. The classifier serves as an evaluation layer to assess whether the learned embeddings capture disease-relevant information.

**Input**
- Sample-level embeddings and corresponding labels:
  - X ∈ R^{2 x d},  y = [0, 1]
    - 0 = cirrhosis
    - 1 = HCC

**Model**

A simple supervised classifier is used, such as:
- Logistic regression
- Linear layer + sigmoid
- Linear SVM

**Output**
- y_hat ∈ {0, 1} or P(HCC | sample)


---

## What connects the modules

The output of Module 3 serves as the input to Module 4, enabling masked self-supervised training. The pretrained encoder from Modules 4–5 is reused in Module 6 to generate spectrum-level embeddings, which are then aggregated in Module 7.
