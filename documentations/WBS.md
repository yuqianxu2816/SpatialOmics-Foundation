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

spectra: List[Spectrum]

Spectrum = {

  "meta": Dict[str, Any],      # e.g., {"PEPMASS": 512.34, "CHARGE": 2, "RTINSECONDS": 1234.5}
  
  "peaks": List[Tuple[float, float]]  # [(mz1, inten1), (mz2, inten2), ...]
  
}

### Step 2 — Extract the (m/z, intensity) pairs from each spectrum
Input

Spectrum["peaks"] = List[(mz, intensity)]
Output

Numeric peak list (possibly cleaned or filtered)

Optional filtering

remove zero or negative intensity

restrict m/z range

keep top-N peaks

Output format

peaks: Array[N, 2]
 column 0 = m/z, column 1 = intensity

### Step 3 — Discretize / tokenize the spectrum (m/z binning)
Input

peaks: Array[N, 2]
Parameters

mz_min, mz_max (e.g., 100–2000)

bin_width (e.g., 1.0 or 0.1 Da)

Output

x: Vector[B]
 aggregated (sum or max) and often normalized intensity per bin

### Step 4 — Randomly mask some peaks
Input

x: Vector[B]
mask_ratio (e.g., 0.15)

Mask strategies

random bins

structured masking (contiguous ranges)

Output

x_masked: Vector[B]
mask:     Vector[B]   # 1 = masked, 0 = visible
targets:  Vector[B]   # ground truth for masked bins
masked bins are set to 0 or a special value

targets are meaningful only where mask = 1

### Step 5 — Train the model to predict masked peaks
This is the self-supervised training step.

Model input

x_masked: Tensor[batch, B]
mask:     Tensor[batch, B]
targets:  Tensor[batch, B]
Model output (training)

y_pred: Tensor[batch, B]
loss:   float
loss computed only on masked positions


### Step 6 — Extract spectrum-level embeddings
After training (or during inference):

Input

unmasked or lightly masked spectrum representation (x or tokenized form)

Output

spectrum_embedding: Vector[d]
fixed-dimensional embedding (e.g., d = 128)

Downstream Tasks

### Step 7 — Aggregate spectrum-level embeddings to sample-level
Input

A set of embeddings from one raw file:

E = {e1, e2, ..., en},  ei in R^d
Output

z in R^d
Aggregation methods

Mean pooling (used in this project)

Median pooling

Attention-based pooling (optional)

sample_embedding: Vector[d]
Note

Due to limited computational resources:

1 cirrhotic raw file -> 1 sample embedding

1 HCC raw file -> 1 sample embedding

This setup is sufficient for validating the proposed pipeline.

### Step 8 — Train a classifier on sample-level embeddings
Input

X in R^{2 x d},  y = [0, 1]
0 = cirrhosis

1 = HCC

Model

Logistic regression

Linear layer + sigmoid

Linear SVM

Output

y_hat in {0, 1}
or
P(HCC | sample)


---

## Problem 2B


### Activity 1: MGF Data Analysis

- **Task:** Implement MGF parser  
- **Deliverable:** Spectrum Object List  
- **Completion:** Capable of correctly reading ≥1 MGF file  


### Activity 2: Masked Self-Supervised Model

- **Task:** Implement the random peak masking strategy  
- **Deliverable:** Trainable self-supervised model  
- **Completion:** Loss is convergent  


### Activity 3: Embedding Evaluation

- **Task:** Compare the distribution of HCC (Hepatocellular Carcinoma) and cirrhosis embeddings  
- **Deliverable:** PCA / UMAP plot  
- **Completion:** There is a separable trend between the two groups


