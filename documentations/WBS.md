## Problem 1B

## Generate MGF Spectra as the First Basic Output

**Objective:**  
Produce MS/MS spectra in MGF format as the first basic computable output for downstream self-supervised learning.

---

### Step 1 — Convert raw LC-MS/MS files to mzML

**Task:**  
Convert vendor-specific `.raw` files into open-format `.mzML` files.

**Actions:**
- Install ProteoWizard
- Use `msconvert` to convert `.raw` → `.mzML`
- Retain full MS1 / MS2 information during conversion

**Deliverable:**  
- One or more valid `.mzML` files corresponding to original raw samples

**Completion criteria:**  
- `.mzML` files can be successfully opened and read by downstream tools (e.g., Pyteomics)

---

### Step 2 — Extract MS2 spectra from mzML

### add the libraries going to use!!! 

**Task:**  
Read `.mzML` files and extract MS/MS (MS2) spectra.

**Actions:**
- Use `pyteomics.mzml` to iterate through spectra
- Filter spectra by `ms level = 2`
- Extract:
  - m/z array
  - intensity array
  - precursor m/z (if available)

**Deliverable:**  
- In-memory representation of MS2 spectra as `(m/z, intensity)` arrays with optional precursor information

**Completion criteria:**  
- Each extracted spectrum contains valid m/z–intensity pairs
- The number of extracted MS2 spectra is greater than zero

---

### Step 3 — Write MS2 spectra to MGF format

**Task:**  
Serialize extracted MS2 spectra into MGF format for model input.

**Actions:**
- Format each spectrum as an MGF block:
  - `BEGIN IONS`
  - precursor information (PEPMASS)
  - `(m/z, intensity)` pairs
  - `END IONS`
- Write spectra using `pyteomics.mgf.write`

**Deliverable:**  
- A `.mgf` file where each `BEGIN IONS … END IONS` block represents one MS/MS spectrum

**Completion criteria:**  
- MGF file is successfully written to disk
- File can be parsed into spectrum objects
- At least one valid spectrum is present in the output MGF


---

## Problem 2B

## Activity 1: MGF Data Inspection and Validation

### Action 1.1 — Load and validate generated MGF files

**Estimated time:** 2 hours

**Sub-tasks:**
- Load existing `.mgf` files generated in Problem 1B
- Iterate through `BEGIN IONS … END IONS` blocks
- Validate required fields (e.g., presence of peak list, optional PEPMASS)
- Check spectrum count and basic structural consistency

**Deliverable:**
- A validated list of spectrum objects loaded from MGF files

**Completion criteria:**
- MGF files can be successfully loaded without errors
- The number of spectra loaded matches expectations
- Each spectrum contains a non-empty peak list

---

### Action 1.2 — Exploratory spectrum statistics and sanity checks

**Estimated time:** 2-3 hours

**Sub-tasks:**
- Verify that m/z and intensity values are numeric
- Remove peaks with zero or negative intensity
- Compute basic statistics:
  - number of peaks per spectrum
  - m/z range distribution
  - intensity distribution
- Optionally restrict m/z range (e.g., 100–2000)

**Deliverable:**
- Cleaned and validated spectrum objects
- Summary statistics describing the MGF data

**Completion criteria:**
- No invalid peaks remain
- Each spectrum contains ≥1 valid peak
- Summary statistics are successfully computed


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
