For each MS/MS spectrum, calculate an embedding vector of a fixed dimension. 

For example: 

input: one MS/MS spectrum (m/z–intensity list) 

output: embedding ∈ R^d (e.g., d = 128) 

Calculation steps:

1. Read the MGF file

2. Extract the (m/z, intensity) pairs from each spectrum

3. Discretize/tokenize the spectrum (e.g., m/z bin)

4. Randomly mask some peaks (masked peaks)

5. Train the model to predict the masked peaks

6. Use the encoder to output the spectrum embedding
