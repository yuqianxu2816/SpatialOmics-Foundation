## Problem 1B

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


---

## Problem 2B

# Problem 2B

## Activity 1: MGF Data Analysis

### Action 1.1 — Implement MGF parser

**Estimated time:** 2–3 hours

**Sub-tasks:**
- Parse `BEGIN IONS` / `END IONS` blocks
- Extract metadata fields (e.g., `PEPMASS`, `CHARGE`, `RTINSECONDS`)
- Parse `(m/z, intensity)` peak pairs
- Handle malformed or empty lines

**Deliverable:**
- A reusable MGF parser function (e.g., `load_mgf()`)
- Output as a list of spectrum objects

**Completion criteria:**
- The parser can correctly read the MGF file giben
- The number of parsed spectra matches the expected count
- Each spectrum object contains the peak list correctly

---

### Action 1.2 — Basic spectrum sanity check

**Estimated time:** 2 hours

**Sub-tasks:**
- Verify m/z and intensity values are numeric
- Remove peaks with zero or negative intensity
- Optionally restrict m/z range (e.g., 100–2000)

**Deliverable:**
- Cleaned spectrum objects ready for preprocessing

**Completion criteria:**
- No invalid peaks remain
- Each spectrum contains ≥1 valid peak


**Total time for Activity 1:** **~4–5 hours**

---

## Activity 2: Masked Self-Supervised Model

### Action 2.1 — Implement random peak masking strategy

**Estimated time:** 2 hours

**Sub-tasks:**
- Define mask ratio (e.g., 15%)
- Randomly select bins or peaks to mask
- Replace masked bins with zeros or a special mask value
- Record mask positions and reconstruction targets
- Use BERT method masking

**Deliverable:**
- A masking function (e.g., `mask_spectrum()`)

**Completion criteria:**
- Masked spectra preserve original shape
- Mask positions and targets are correctly aligned

---

### Action 2.2 — Implement self-supervised reconstruction model

**Estimated time:** 3–4 hours

**Sub-tasks:**
- Define a simple encoder–decoder architecture  
  (e.g., MLP or lightweight Transformer)
- Implement forward pass from masked input to reconstructed output
- Define loss function computed only on masked bins

**Deliverable:**
- A trainable self-supervised model class

**Completion criteria:**
- Model can run forward and backward passes
- Training loss decreases over iterations

---

### Task 2.3 — Self-supervised pretraining

**Estimated time:** 3-4 hours

**Sub-tasks:**
- Prepare mini-batches of masked spectra
- Train the model for a small number of epochs
- Monitor reconstruction loss during training

**Deliverable:**
- A pretrained encoder capable of generating embeddings

**Completion criteria:**
- Loss curve shows a convergent trend
- No runtime or memory errors occur during training

**Total time for Activity 2:** **8–10 hours**

---

## Activity 3: Embedding Evaluation

### Action 3.1 — Extract spectrum-level embeddings

**Estimated time:** 1–2 hours

**Sub-tasks:**
- Freeze or reuse the pretrained encoder
- Pass unmasked spectra through the encoder
- Collect spectrum-level embedding vectors

**Deliverable:**
- A matrix of spectrum embeddings (`n_spectra × d`)

**Completion criteria:**
- Each spectrum has a corresponding embedding vector
- Embedding dimensionality is consistent (e.g., `d = 128`)

---

### Action 3.2 — Aggregate embeddings to sample level

**Estimated time:** 1 hour

**Sub-tasks:**
- Group spectrum embeddings by raw file (sample)
- Apply mean pooling as the primary aggregation method
- Optionally compare with median pooling

**Deliverable:**
- One sample-level embedding per raw file

**Completion criteria:**
- Exactly one embedding for cirrhosis and one for HCC
- Aggregation method is deterministic and reproducible

---

### Action 3.3 — Visual embedding evaluation

**Estimated time:** 1–2 hours

**Sub-tasks:**
- Apply PCA or UMAP to sample-level embeddings
- Generate a 2D visualization
- Label points by disease status

**Deliverable:**
- PCA or UMAP plot comparing HCC and cirrhosis embeddings

**Completion criteria:**
- Plot is generated without errors
- A separable or distinguishable trend is observable between the two groups

**Total time for Activity 3:** **~3–5 hours**

---

## Overall Time Estimate

| Activity | Estimated Time |
|--------|----------------|
| Activity 1: MGF Data Analysis | 4–5 hours |
| Activity 2: Masked Self-Supervised Model | 8–10 hours |
| Activity 3: Embedding Evaluation | 3–5 hours |
| **Total** | **~15–20 hours** |
