- BN-visualization/

  Visualization of structural brain network, cross-modal brain network, and functional brain network for each subject with file name corresponding to subject index number (0-305).

- brain network construction

  - constructFBN_PC.py

    Modules for constructing functional brain networks using Pearson's correlation coefficient (PC).

  - constructFBN_ot.py

    Modules for constructing cross-modal brain networks using optimal transport (ot).

- model

  - model.py

    The network for classification constructed by GCN and Siamese network.

  - main.py

    Training and testing file.