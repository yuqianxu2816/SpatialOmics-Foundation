## Problem 1B

For each MS/MS spectrum, calculate an embedding vector of a fixed dimension.

**Example:**

- **Input:** one MS/MS spectrum (m/z–intensity list)  
- **Output:** embedding ∈ R^d (e.g., d = 128)

### Calculation Steps

1. Read the MGF file  
2. Extract the (m/z, intensity) pairs from each spectrum  
3. Discretize / tokenize the spectrum (e.g., m/z bin)  
4. Randomly mask some peaks (masked peaks)  
5. Train the model to predict the masked peaks  
6. Use the encoder to output the spectrum embedding  


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
