# Breast Cancer Classification with Quantum Kernel Fusion

We tested whether a quantum approach to comparing cell measurements could improve breast cancer classification. We kept the same data, preprocessing and type of SVM classifier, then compared a classical RBF kernel with a simulated quantum fidelity kernel. The two approaches also use different ways of combining feature groups, which we explain below.

Our dataset contains **569 records**: 398 for training, 57 for choosing model settings, and 114 for final testing. Those 114 test records include **72 noncancerous and 42 cancerous cases**. Both models use exactly the same split.

**The quantum model correctly classified 111 of the 114 test records (97.37% accuracy), compared with 108 of 114 (94.74%) for the classical model.** Within the 42 cancerous cases in that test set, quantum correctly identified 41 and classical identified 39.

This repository contains our original experiment: the notebook, dataset, results and instructions to run it.

## Why we chose this problem

Our task is to classify a breast mass as **benign (noncancerous)** or **malignant (cancerous)** using measurements of cell nuclei. We chose a public, labelled dataset so we could build and reproduce both approaches under the same data conditions.

We also chose this topic in recognition of Pinktober, Breast Cancer Awareness Month. Working on the project during October gave us another reason to focus on breast cancer awareness and explore how quantum machine learning could help classify breast masses as cancerous or noncancerous.

For this task, we wanted to look beyond accuracy. A model can have a high overall score and still miss cancerous cases. We therefore report recall, precision and F1 alongside accuracy, and show the actual mistakes each model made.

The inputs are measurements already extracted from fine-needle aspirate images. Our code works with these measurements; it does not read medical images directly. This is a research comparison, not a clinical diagnostic tool.

## Our results

We selected each model's settings using the validation records, then evaluated it on the same separate test records.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Classical fusion SVM | 94.74% | 92.86% | 92.86% | 92.86% |
| Quantum fusion SVM | **97.37%** | **95.35%** | **97.62%** | **96.47%** |

The quantum approach improved accuracy by **2.63 percentage points** and F1 by **3.61 percentage points** on this split. These differences use the unrounded scores.

Here is what those percentages mean in records. The test set contains 72 noncancerous and 42 cancerous cases:

| Test outcome | Classical | Quantum |
|---|---:|---:|
| Noncancerous cases correctly identified | 69 | 70 |
| Noncancerous cases incorrectly flagged as cancerous | 3 | 2 |
| Cancerous cases correctly identified | 39 | 41 |
| Cancerous cases missed | 3 | 1 |
| Total correct predictions | 108 / 114 | 111 / 114 |

For precision, recall and F1, **cancerous is the positive class**, encoded as `1`. Noncancerous is encoded as `0`.

- **Accuracy:** how often the model is correct across both classes.
- **Precision:** of the cases called cancerous, how many really are cancerous.
- **Recall:** of all cancerous cases, how many the model finds.
- **F1:** a score balancing precision and recall. We use validation F1 to choose settings.

The results support a performance improvement for this experiment. Because the test set is small and comes from one split, they do not establish that the same improvement will hold on every new dataset.

## Dataset and split

We use the [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), originally provided through the UCI Machine Learning Repository. It contains **569 records and 30 input measurements**, with 357 benign and 212 malignant cases.

The measurements describe cell size, shape and texture. They are organised into three groups of ten: `mean`, `se` (standard error), and `worst` (the mean of the three largest values for each measurement, as described in the dataset documentation).

We split the records as follows:

| Part | Records | Purpose |
|---|---:|---|
| Training | 398 | Learn preprocessing and fit the models |
| Validation | 57 | Choose model settings |
| Test | 114 | Calculate the final reported scores |
| **Total** | **569** | **Complete dataset** |

This is approximately a 70% / 10% / 20% split. Both splitting steps are stratified by diagnosis and use `random_state=10`. Both models use all 398 training records.

## How we prepare the data

We remove the record ID and the completely empty `Unnamed: 32` column. The diagnosis is kept as the target, not an input. The notebook replaces infinite values and drops remaining incomplete rows; none of the supplied measurement rows are lost. It also fits a median imputer on training data as a safeguard.

We then:

1. Standardize the 30 measurements using the training records.
2. Apply PCA separately to the mean, SE and worst groups, reducing each from ten measurements to four components.
3. Standardize the four PCA scores in each group.
4. Apply those fitted transformations to validation and test records.

All fitted preprocessing comes from training data only. The correlation plots are exploratory; we do not use full-dataset label correlations to select features.

## Our classical baseline

We use an **RBF kernel fusion SVM**. An RBF kernel gives a similarity score based on the distance between two feature vectors. It provides a nonlinear baseline while keeping the final classifier and PCA inputs comparable to the quantum approach.

We calculate one RBF kernel for each feature group and combine them with equal weights:

`classical kernel = (mean kernel + SE kernel + worst kernel) / 3`

The result goes into scikit-learn's `SVC(kernel="precomputed", class_weight="balanced")`.

We try six values of `C` and four values of `gamma`, giving **24 configurations**. Validation F1 selects `C=0.1` and `gamma=0.1`.

## Our quantum contribution

The quantum model changes how similarity is calculated. We encode each group's four PCA components into a **four-qubit feature map**, calculate the overlap between encoded states, and pass the resulting kernel to the same type of classical SVM.

For each feature group, the feature map represents:

1. Hadamard preparation of a uniform state.
2. RZ rotations whose angles depend on individual input values.
3. RZZ interactions whose angles depend on products of pairs of input values.

We calculate the fidelity kernel as:

$$
K_Q(x,y)=|\langle\psi(x)\mid\psi(y)\rangle|^2.
$$

The three group kernels are combined with nonnegative weights that sum to one. Unlike the classical baseline, the quantum approach chooses these weights using validation F1. A weight of zero lets it leave out a group.

### Why this design

We wanted to test an interaction-dependent alternative to the baseline's distance-based similarity. Pairs of PCA features affect quantum phases through RZZ terms. Four components keep each group circuit small, while fusion combines the measurement families.

RBF kernels can also model nonlinear relationships. We test whether our quantum feature map and fusion design improve classification on the same inputs.

We calculate the states and fidelities exactly using **NumPy statevector simulation**. The final classifier is a classical SVM. This is a hybrid quantum-classical approach; no quantum hardware, measurement shots or variational circuit training is used.

## Model settings

| Setting | Classical search | Quantum search |
|---|---|---|
| SVM `C` | 0.01, 0.1, 1, 10, 100, 1000 | Same values |
| RBF `gamma` | 0.1, 0.25, 0.5, 1.0 | Not applicable |
| Encoding angle range | Not applicable | 0 to π, 2π or 4π |
| Qubit connections | Not applicable | Linear or all-to-all |
| Interaction strength | Not applicable | 0.25, 0.5, 1.0 |
| Group weights | Equal: 1/3 each | Multiples of 0.25, summing to one |
| Total configurations | 24 | 1,620 |

There are 15 quantum weight combinations. Both searches keep the first setting in grid order when validation F1 is tied. The quantum angle scalers are fitted on training PCA scores, with clipping for values outside the fitted range.

| Selected setting | Classical | Quantum |
|---|---|---|
| SVM `C` | 0.1 | 100 |
| RBF `gamma` | 0.1 | Not applicable |
| Maximum encoding angle | Not applicable | π |
| Qubit connections | Not applicable | Linear: (0,1), (1,2), (2,3) |
| Interaction strength | Not applicable | 0.25 |
| Mean / SE / worst weights | 1/3 / 1/3 / 1/3 | 0 / 0.25 / 0.75 |
| Validation F1 | 97.67% | 100.00% |

The selected quantum model uses the SE and worst kernels; the mean group receives zero weight. All three group kernels are constructed during the search.

## Resources and efficiency

| Resource | Quantum approach |
|---|---|
| Qubits per feature-group circuit | 4, reusable across groups |
| Statevector per record and group | 16 complex amplitudes |
| Selected feature-map operations | 4 Hadamard, 4 RZ and 3 RZZ |
| Execution | Exact NumPy simulation on a classical computer |
| Hardware shots | Not used |

The gate counts describe one selected feature-map state preparation, not a complete hardware circuit for estimating fidelity. The training kernel is **398 × 398** and the test-to-training kernel is **114 × 398** for both approaches. Larger datasets would therefore increase the kernel storage and computation required by both models.

We also measured the original classical and quantum search cells in one reproduction run:

| Model | Search and test-prediction time |
|---|---:|
| Classical fusion SVM | About 0.11 seconds |
| Quantum fusion SVM | About 9.99 seconds |

These timings include kernel construction, the complete parameter search, candidate SVM fitting and the selected test predictions. Shared preprocessing and plotting are excluded. The run used Python 3.12.14 on an x86-64 CPU with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. They are approximate times from one run, not averages, and will vary by machine.

The quantum search tries more configurations and takes longer in this simulation. The observed advantage is predictive performance on this split; we do not claim a quantum speed advantage.

## How fair is the comparison?

Both models share the same data partitions, training records, preprocessing, feature groups, PCA dimensions, balanced SVM class weighting and validation-F1 selection rule. The test labels are not used by the code to select settings.

The quantum approach tunes its group weights and searches more settings; the classical approach uses equal weights. We report the gain for the **complete quantum fusion approach**. This experiment does not isolate how much comes from the circuit, group weighting or larger search.

The original notebook reports one split. Its 57-record validation set is small for choosing among 1,620 quantum configurations, and 100% validation F1 is not proof of perfect generalization or no overfitting. A few predictions can change the test scores noticeably.

Useful follow-up work would repeat the evaluation across different splits, compare matched fusion options and search budgets, test an interaction-free feature map, and evaluate independent data and hardware noise. This experiment does not establish clinical readiness or computational quantum advantage.

## Run the project

The original notebook uses Google Colab's CSV-upload function.

1. Open [Google Colab](https://colab.research.google.com/) and open `breast_cancer_classical_quantum.ipynb` from GitHub, or upload a downloaded copy.
2. Upload `requirements.txt` through Colab's Files sidebar.
3. To use the package versions from the reproduction checks, run this in a setup cell before the notebook:

   ```python
   !pip install -r /content/requirements.txt
   ```

4. Restart the runtime if requested after installation, then run the notebook cells in order. A CPU runtime is enough for this code.
5. When the first cell opens the file picker, select `breast_cancer.csv`.
6. Read the selected settings and final comparison table printed by the notebook.

The reproduction checks used Python 3.12.14 and the versions in `requirements.txt`. NumPy handles the quantum simulation, so no separate quantum package is needed. Colab provides `google.colab`; `pickle` and `itertools` come with Python.

For cell runtime measurements, put `from time import perf_counter` and `started = perf_counter()` at the start of a model-search cell, and `print(perf_counter() - started)` at its end. This measures the scope used in the timing table.

## Files in this repository

| File | What it contains |
|---|---|
| `breast_cancer_classical_quantum.ipynb` | Our original data preparation, models and evaluation |
| `breast_cancer.csv` | The 569-record input dataset |
| `results.csv` | Original test scores as percentages, with dataset and split sizes |
| `requirements.txt` | Package versions used for the reproduction checks |
| `README.md` | Project explanation, methods, results and instructions |

In `results.csv`, a value of `97.37` under `Accuracy (%)` means 97.37% accuracy. Precision, recall and F1 refer to cancerous cases. The CSV is a summary of the reported scores; the notebook calculates the results. Run the notebook to generate the comparison table.

Running the notebook also creates processed data/label CSVs for the three splits and `cancer_imputer_scaler.pkl`. These are generated outputs, not extra input datasets.

## Sources and credit

- **Dataset:** [Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Credit to the dataset's creators, Wolberg, Mangasarian and Street. UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **SVM implementation:** [scikit-learn SVC](https://scikit-learn.org/1.8/modules/generated/sklearn.svm.SVC.html).
- **Preprocessing:** [scikit-learn PCA](https://scikit-learn.org/1.8/modules/generated/sklearn.decomposition.PCA.html) and [RBF kernel](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html).
- **Quantum-kernel background:** Havlíček et al., [Supervised learning with quantum enhanced feature spaces](https://arxiv.org/abs/1804.11326). This is background on the method family; our notebook does not reproduce that paper's hardware experiment or exact circuit.

We use scikit-learn for the SVMs, PCA and RBF kernels, and NumPy for our simulated quantum states and fidelity kernels.
