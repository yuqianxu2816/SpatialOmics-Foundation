# SpatialOmics-Foundation

1. Problems solved
Although single-cell RNA sequencing can resolve cell types and states, it cannot retain spatial positional information and cellular neighborhood structure due to the need for tissue dissociation. This makes it difficult to address the following issues:

Where is the true spatial position of cells in the tissue?
What cells are surrounding it? What is the proportion?
Which organizational region or functional microenvironment (niche) does it belong to?
How to transfer the structural knowledge of spatial transcriptomics to traditional scRNA-seq data?

Therefore, a unified model that can simultaneously learn from both single-cell and spatial omics data is needed, capable of capturing spatial patterns across tissues, platforms, and species.

2. Method used
To address the aforementioned issues, the following methods were adopted:

(1) Convert cell expression into serialized input:
Construct gene token sequence using gene expression ranking.
Add contextual tokens such as modality, species, or technology.
Enable the model to generalize across platforms and organizations.

(2) Self-supervised pre-training based on Transformer:
Randomly mask and predict gene tokens in the sequence.
Learn expression relationships and structural patterns from large-scale cellular data.
Obtain a unified cell embedding.

(3) No need to integrate data in advance:
Directly retain the differences between different technologies and organizations.
Utilize models to automatically learn cross-modal associations.

3. Input and output (at the single-cell level)
Input:
Gene-rank tokens
Relevant contextual tokens (such as technology, modality, species, etc.)

Output:
A low-dimensional embedding (cell representation)
Incorporating spatial context
It can be used for any downstream prediction tasks

4. Downstream tasks
Using embeddings, various spatially related tasks can be executed, including:

(1) Spatial label prediction:
cell type
Spatial niche
Tissue regions (such as cortical layers, liver zones, etc.)

(2) Neighborhood composition prediction:
Within a fixed radius, the proportion of each cell type
It can reflect organizational structure and local signal environment

(3) Neighborhood density prediction:
Predict local cell density
It aids in identifying high-density areas, such as the tumor microenvironment

(4) Transfer spatial information to non-spatial data:
Inferring the most probable spatial locations for scRNA-seq data
Enrich traditional single-cell atlas with spatial knowledge
