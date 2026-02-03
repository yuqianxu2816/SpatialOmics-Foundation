## 版本1 ##:


### Objective

For each MS/MS spectrum, calculate an embedding vector of a fixed dimension.

### Example

- Input: one MS/MS spectrum (m/z–intensity list)
- Output: embedding in R^d (e.g., d = 128)


---

## Calculation Steps


### Step 1 — Read the MGF file

**Input**

- File: `*.mgf`
- Format: plain text file containing multiple blocks:

BEGIN IONS

m/z intensity

m/z intensity

...

END IONS

**Output**

- A list of spectrum objects (one per BEGIN IONS ... END IONS block)

**Output format (Python-like)**

- spectra: List[Spectrum]

Spectrum = {

  "meta": Dict[str, Any],      # e.g., {"PEPMASS": 512.34, "CHARGE": 2, "RTINSECONDS": 1234.5}
  
  "peaks": List[Tuple[float, float]]  # [(mz1, inten1), (mz2, inten2), ...]
  
}

### Step 2 — Extract the (m/z, intensity) pairs from each spectrum

**Input**

- One Spectrum object from Step 1

- Spectrum["peaks"] = List[(mz, intensity)]

**Output**

- Numeric peak list (possibly cleaned or filtered)

**Optional filtering**

- remove zero or negative intensity

- restrict m/z range

- keep top-N peaks

**Output format**

- peaks: Array[N, 2]
 - column 0 = m/z, column 1 = intensity

### Step 3 — Discretize / tokenize the spectrum (m/z binning)

**Input**

- peaks: Array[N, 2]
- Binning parameters:
  - mz_min, mz_max (e.g., 100–2000)
  - bin_width (e.g., 1.0 or 0.1 Da)

**Output**
- A fixed-length intensity vector x of length B (number of bins)
  
**Output format**
- x: Vector[B]
- aggregated (sum or max) and often normalized intensity per bin

### Step 4 — Randomly mask some peaks

**Input**

- Binned vector x: Vector[B]
- mask_ratio (e.g., 0.15)

- Mask strategies
  - random bins
  - structured masking (contiguous ranges)

**Output**
- x_masked: Vector[B] (masked input)
- mask: Vector[B] (binary indicator: 1 = masked bin)
- targets: Vector[B] (ground truth values for masked bins only)


### Step 5 — Train the model to predict masked peaks
This is the self-supervised training step.

**Model input**

- x_masked (or masked token sequence): Tensor[batch, B]
- mask (to compute loss only on masked locations): Tensor[batch, B]
- targets: Tensor[batch, B]
  
**Model output (training)**
- y_pred: reconstructed intensities (or token logits) over the full vector/sequence
- loss: computed on masked positions only

**output format**
- y_pred: Tensor[batch, B]
- loss:   float


### Step 6 — Extract spectrum-level embeddings
After training (or during inference):

**Input**

- unmasked or lightly masked spectrum representation (x or tokenized form)
  - x: Vector[B] or (token_ids, token_values, attention_mask)

**Output**

- A spectrum embedding vector:
  - embedding ∈ R^d (e.g., d = 128)

### Downstream Tasks

### Step 7 — Aggregate spectrum-level embeddings to sample-level
**Input**

- A set of embeddings from one raw file: (Each embedding corresponds to one MS/MS spectrum from the same sample.)
  - E = {e1, e2, ..., en},  ei ∈ R^d

**Output**
- A single sample-level embedding:
  - z ∈ R^d
**Aggregation methods**

- Mean pooling (used in this project)

- Median pooling

- Attention-based pooling (optional)
  
**Output format**
- sample_embedding: Vector[d]
  
**Note**

Due to limited computational resources:

1 cirrhotic raw file -> 1 sample embedding

1 HCC raw file -> 1 sample embedding

This setup is sufficient for validating the proposed pipeline.

### Step 8 — Train a classifier on sample-level embeddings
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


## 版本2 ##


## Module 1: Data I/O

### Function
- Read `.mgf` files  
- Parse spectra and associated metadata  

### Output
- List of spectra in Python object format  


---

## Module 2: Spectrum Preprocessing

### Function
- m/z binning  
- Intensity normalization  
- Peak number pruning / padding  

### Output
- Token sequence usable by the model  


---

## Module 3: Masked Self-supervised Learning

### Function
- Randomly mask a subset of peaks  
- Define reconstruction objectives  

### Model
- Transformer or MLP encoder  
  *(a simplified version is sufficient for initial experiments)*  


---

## Module 4: Embedding Extraction

### Function
- Output spectrum-level embeddings  
- Aggregate into sample-level embeddings  
  - Mean pooling  
  - Attention-based pooling  


---

## Module 5: Downstream Analysis

### Function
- HCC vs. cirrhosis classification  
- Embedding visualization (PCA / UMAP)  
