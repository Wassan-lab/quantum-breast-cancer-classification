# Breast Cancer Classification with Quantum Kernel Fusion

We built a breast cancer classification project to explore how quantum machine learning could improve predictions from cell measurements. Our approach combines four-qubit quantum feature maps with a support vector machine (SVM). We compare its results with a classical radial basis function (RBF) kernel fusion SVM.

We use **569 records in total**: **398 for training, 57 for choosing model settings, and 114 for final testing**. Both models use the same records in each part. The 114 test records contain 72 noncancerous and 42 cancerous cases.

**Our quantum model correctly classified 111 of the 114 test records, reaching 97.37% accuracy and 96.47% F1. The classical model correctly classified 108, reaching 94.74% accuracy and 92.86% F1.** Among the 42 cancerous test cases, quantum identified 41 and classical identified 39.

Our project brings together three ideas:

- **A healthcare task with clear outcomes:** classify breast masses using measured cell features and track cancerous cases found and missed.
- **Small quantum feature maps:** use four qubits per feature group after reducing the input dimensions with principal component analysis (PCA).
- **Feature-group fusion:** calculate similarities for the mean, standard-error and worst measurements, then combine them for classification.

## Why we chose this problem

Our task is to classify a breast mass (lump) as **benign, meaning noncancerous**, or **malignant, meaning cancerous**. We work with measurements of cell nuclei extracted from fine-needle aspirate images of breast masses.

We also chose this topic in recognition of Pinktober, Breast Cancer Awareness Month. Working on the project during October gave us another reason to focus on breast cancer awareness and explore how quantum machine learning could help classify breast masses as cancerous or noncancerous.

We wanted our evaluation to show more than an overall accuracy score. Finding cancerous cases matters to this task, so we compare recall, precision and F1 as well as accuracy. We use a public, labelled dataset so that both approaches can be reproduced and their predictions checked against known diagnoses.

The project is an experimental classification study using existing cell measurements. The notebook takes these measurements as its inputs.

## Our results

We choose each model's settings using validation F1, then calculate the final scores on the same 114 test records.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Classical fusion SVM | 94.74% | 92.86% | 92.86% | 92.86% |
| Quantum fusion SVM | **97.37%** | **95.35%** | **97.62%** | **96.47%** |

The quantum fusion approach scores higher on all four metrics in this experiment. The improvement is **2.63 percentage points in accuracy** and **3.61 percentage points in F1**, calculated from the unrounded scores.

The prediction counts show where this improvement comes from:

| Test outcome | Classical | Quantum |
|---|---:|---:|
| Noncancerous cases correctly identified | 69 | 70 |
| Noncancerous cases incorrectly flagged as cancerous | 3 | 2 |
| Cancerous cases correctly identified | 39 | 41 |
| Cancerous cases missed | 3 | 1 |
| Total correct predictions | 108 / 114 | 111 / 114 |

Quantum finds two additional cancerous cases and makes one fewer false alarm, giving three more correct predictions overall.

### What the metrics mean

In our code, cancerous cases are labelled `1` and noncancerous cases are labelled `0`. Precision, recall and F1 focus on the cancerous class.

| Metric | What it tells us |
|---|---|
| Accuracy | How often the model is correct across both classes |
| Precision | Of the cases called cancerous, how many really are cancerous |
| Recall | Of all cancerous cases, how many the model finds |
| F1 | A score balancing precision and recall; our metric for choosing settings |

## Our dataset and split

We use the [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) from the UCI Machine Learning Repository. It contains **569 records, 30 input measurements, 357 benign cases and 212 malignant cases**.

The measurements describe properties such as cell size, shape and texture. They are organised into three groups of ten:

- **Mean:** mean values of the measurements.
- **SE:** their standard errors.
- **Worst:** the mean of the three largest values for each measurement, following the dataset's definition.

| Part | Records | What we use it for |
|---|---:|---|
| Training | 398 | Fit preprocessing and train both models |
| Validation | 57 | Choose model settings using F1 |
| Test | 114 | Calculate the final reported scores |
| **Total dataset** | **569** | **All available records** |

The split is approximately 70% training, 10% validation and 20% testing. Both splitting steps use `random_state=10` and stratification by diagnosis to maintain class proportions. Both approaches use all 398 training records.

## Preparing the measurements

We remove the record ID and the empty `Unnamed: 32` column, and keep diagnosis as the target. The notebook replaces infinite values and removes remaining incomplete rows. All 569 records remain for the supplied dataset. A median imputer is also fitted on training data.

We then:

1. Standardize the 30 measurements using the training records.
2. Apply PCA separately to each feature group, reducing ten measurements to four components.
3. Standardize each group's PCA scores.
4. Transform validation and test records using the fitted training objects.

This gives both approaches the same three groups of four PCA components. All fitted preprocessing comes from training data only. The correlation plots help us inspect the data; feature inclusion follows the three predefined measurement groups.

## Our classical model

Our baseline is an **RBF kernel fusion SVM**. The RBF kernel calculates similarity from the distance between feature vectors and gives us a nonlinear classical comparison using the same PCA inputs and type of final classifier.

We calculate an RBF kernel for each feature group and give the three kernels equal weight:

`classical kernel = (mean kernel + SE kernel + worst kernel) / 3`

The combined kernel goes into scikit-learn's `SVC(kernel="precomputed", class_weight="balanced")`. We try six `C` values and four `gamma` values, giving **24 configurations**. Validation F1 selects `C=0.1` and `gamma=0.1`.

## Our quantum model

Our quantum contribution is a different similarity function, built from the overlap between encoded quantum states. We encode each group's four PCA components in a **four-qubit feature map**, calculate its fidelity kernel, and combine the group kernels for an SVM classifier.

Each feature map represents:

1. **Hadamard preparation:** start with a uniform state.
2. **RZ rotations:** encode individual input values as phases.
3. **RZZ interactions:** encode products of pairs of input values through two-qubit interactions.

For two input records, the group similarity is:

$$
K_Q(x,y)=|\langle\psi(x)\mid\psi(y)\rangle|^2.
$$

We combine the three group kernels with nonnegative weights that sum to one. Validation F1 selects these weights together with the circuit settings and SVM `C`. Zero weights let the model leave out a measurement group.

### Why we used this design

The RZZ terms make pairs of PCA features contribute to quantum phases. This gives us an interaction-dependent alternative to the RBF baseline's distance-based similarity. Four PCA components keep each circuit at four qubits, while fusion lets us combine information from the three measurement families.

The classical and quantum approaches both model nonlinear relationships. Our experiment tests how this particular quantum feature map and fusion strategy perform on the same prepared inputs.

We calculate the quantum states and fidelities exactly with **NumPy statevector simulation**, then train a classical SVM on the combined kernel. This is a hybrid quantum-classical implementation.

## Search settings and selected models

| Setting | Classical search | Quantum search |
|---|---|---|
| SVM `C` | 0.01, 0.1, 1, 10, 100, 1000 | Same values |
| RBF `gamma` | 0.1, 0.25, 0.5, 1.0 | Not applicable |
| Encoding angle range | Not applicable | 0 to π, 2π or 4π |
| Qubit connections | Not applicable | Linear or all-to-all |
| Interaction strength | Not applicable | 0.25, 0.5, 1.0 |
| Group weights | Equal: 1/3 each | Multiples of 0.25, summing to one |
| Total configurations | 24 | 1,620 |

The quantum search includes 15 weight combinations. Each model keeps the first setting in grid order when validation F1 is tied. Quantum angle scalers are fitted on training PCA scores, with clipping for values outside the fitted range.

| Selected setting | Classical | Quantum |
|---|---|---|
| SVM `C` | 0.1 | 100 |
| RBF `gamma` | 0.1 | Not applicable |
| Maximum encoding angle | Not applicable | π |
| Qubit connections | Not applicable | Linear: (0,1), (1,2), (2,3) |
| Interaction strength | Not applicable | 0.25 |
| Mean / SE / worst weights | 1/3 / 1/3 / 1/3 | 0 / 0.25 / 0.75 |
| Validation F1 | 97.67% | 100.00% |

The selected quantum model gives 25% weight to the SE kernel and 75% to the worst kernel. All three group kernels are constructed during the search; the selected mean weight is zero.

## Evaluation setup

Both approaches share the same records, split, training-only preprocessing, feature groups, PCA dimensions, balanced SVM class weighting and validation-F1 selection rule. The code uses validation records to choose settings and the separate test records to calculate the reported performance.

Our comparison measures the complete fusion approaches. The classical model uses equal group weights, while the quantum model selects group weights alongside its circuit settings. Their search sizes are listed above. The reported gain measures the full design, with the circuit, group fusion and tuning choices evaluated together.

These results come from the stated stratified split and exact simulated kernels. Repeating the study across data splits and on independent data would help assess how consistently the improvement carries over.

## Quantum resources and runtime

| Resource | Quantum implementation |
|---|---|
| Qubits per feature-group circuit | 4, reusable across groups |
| Statevector per record and group | 16 complex amplitudes |
| Selected state-preparation operations | 4 Hadamard, 4 RZ and 3 RZZ |
| Training kernel shape | 398 × 398, also used by the classical approach |
| Test-to-training kernel shape | 114 × 398, also used by the classical approach |
| Execution | Exact NumPy simulation on a classical CPU |
| Hardware shots | Not used |

The operation counts describe one feature-map state preparation. A hardware circuit for estimating fidelity would have additional requirements. The original implementation evaluates quantum states directly through NumPy.

One reproduction run gave the following timings for the complete model-search cells:

| Model | Configurations | Search and test-prediction time |
|---|---:|---:|
| Classical fusion SVM | 24 | About 0.11 seconds |
| Quantum fusion SVM | 1,620 | About 9.99 seconds |

These times include kernel construction, candidate SVM fitting, the full search and the selected test predictions. Shared preprocessing and plotting are excluded. The run used Python 3.12.14 on an x86-64 CPU with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. These are single-run CPU measurements and will vary by machine.

We report configuration counts with runtime so the cost of each search can be read alongside its classification results.

## Run our project

1. Open [Google Colab](https://colab.research.google.com/) and open `breast_cancer_classical_quantum.ipynb` from GitHub, or upload a downloaded copy.
2. Upload `requirements.txt` through Colab's Files sidebar.
3. To use the versions from our reproduction checks, run this setup cell before running the notebook:

   ```python
   !pip install -r /content/requirements.txt
   ```

4. Restart the runtime if requested after installation. A CPU runtime is enough for the code.
5. Run the notebook cells in order. When the first cell opens the file picker, choose `breast_cancer.csv`.
6. The notebook prints the selected settings and displays the final results table.

The reproduction checks used Python 3.12.14 and the versions in `requirements.txt`. Colab provides the CSV-upload interface. NumPy handles the quantum simulation, and no separate quantum package is required.

To reproduce the cell timing scope, put `from time import perf_counter` and `started = perf_counter()` at the start of each model-search cell, and `print(perf_counter() - started)` at its end.

## Files in this repository

| File | Contents |
|---|---|
| `breast_cancer_classical_quantum.ipynb` | Original data preparation, classical model, quantum model and evaluation |
| `breast_cancer.csv` | Our 569-record input dataset |
| `results.csv` | Original test scores as percentages, with the dataset and split sizes |
| `requirements.txt` | Package versions used for the reproduction checks |
| `README.md` | Project explanation, methods, results and run instructions |

In `results.csv`, `97.37` under `Accuracy (%)` means 97.37% accuracy. Precision, recall and F1 refer to cancerous cases. The notebook calculates the results; the CSV provides a summary for easy comparison.

The notebook also saves processed data and label CSVs for each split, plus `cancer_imputer_scaler.pkl`. These files are generated when the notebook runs.

## Sources and credit

- **Dataset:** [Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Credit to the dataset's creators, Wolberg, Mangasarian and Street. The dataset is listed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Classifiers and preprocessing:** scikit-learn's [SVC](https://scikit-learn.org/1.8/modules/generated/sklearn.svm.SVC.html), [PCA](https://scikit-learn.org/1.8/modules/generated/sklearn.decomposition.PCA.html) and [RBF kernel](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html) implementations.
- **Quantum-kernel background:** Havlíček et al., [Supervised learning with quantum enhanced feature spaces](https://arxiv.org/abs/1804.11326), for background on quantum feature spaces and kernel methods.

We use these established methods to build and evaluate our feature-group fusion approach, with NumPy implementing the simulated quantum states and fidelity kernels.
