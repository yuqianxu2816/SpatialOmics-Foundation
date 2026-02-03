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
