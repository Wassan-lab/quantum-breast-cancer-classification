# Breast Cancer Classification with Quantum Kernel Fusion

We built a breast cancer classification project to explore how quantum machine learning could improve predictions from cell measurements. Our approach uses IBM Qiskit to build four-qubit quantum feature maps and combines their fidelity kernels with a support vector machine (SVM). We compare its results with a classical radial basis function (RBF) kernel fusion SVM.

We use **569 records in total**: **398 for training, 57 for choosing model settings, and 114 for final testing**. Both models use the same records in each part. The 114 test records contain 72 noncancerous and 42 cancerous cases.

**In our earlier custom-statevector run, the quantum model correctly classified 111 of the 114 test records, reaching 97.37% accuracy and 96.47% F1. The classical model correctly classified 108, reaching 94.74% accuracy and 92.86% F1.** Among the 42 cancerous test cases, quantum identified 41 and classical identified 39. We have now switched the quantum circuit implementation to Qiskit and reduced its search to 24 configurations. The Qiskit results still need to be measured by running the updated notebook.

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

We choose each model's settings using validation F1, then calculate the final scores on the same 114 test records. The scores below are from our earlier custom-statevector implementation, not a completed Qiskit run.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Classical fusion SVM | 94.74% | 92.86% | 92.86% | 92.86% |
| Quantum fusion SVM | **97.37%** | **95.35%** | **97.62%** | **96.47%** |

In the earlier custom-statevector experiment, the quantum fusion approach scored higher on all four metrics. The improvement is **2.63 percentage points in accuracy** and **3.61 percentage points in F1**, calculated from the unrounded scores.

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

We now build the circuits using **Qiskit `QuantumCircuit`** and calculate exact statevectors using **Qiskit `Statevector`**. NumPy calculates the state overlaps and fidelity kernels, then scikit-learn trains an SVM on the combined kernel. This is still a hybrid quantum-classical implementation, simulated on a classical computer rather than run on IBM quantum hardware.

## Search settings and selected models

We now test **24 configurations for each model**. The classical grid stays the same. For the Qiskit model, we keep the circuit settings that performed best in our earlier search and compare four fusion-weight combinations across six SVM `C` values.

| Setting | Classical search | Qiskit quantum search |
|---|---|---|
| SVM `C` | 0.01, 0.1, 1, 10, 100, 1000 | Same values |
| RBF `gamma` | 0.1, 0.25, 0.5, 1.0 | Not applicable |
| Encoding angle range | Not applicable | 0 to π |
| Qubit connections | Not applicable | Linear: (0,1), (1,2), (2,3) |
| Interaction strength | Not applicable | 0.25 |
| Group weights (mean / SE / worst) | Equal: 1/3 each | (0, 0.25, 0.75), (0, 0.5, 0.5), (0, 0, 1), (0.25, 0.25, 0.5) |
| Total configurations | **24** | **24** |

The classical search tests six `C` values and four `gamma` values (6 × 4 = 24). The quantum search tests six `C` values and four group-weight combinations (6 × 4 = 24). Both choose the setting with the highest validation F1, keeping the first candidate when scores are tied.

The earlier custom quantum model selected `C=100` and weights `(0, 0.25, 0.75)`, with 100% validation F1 in that run. We included those weights in the new grid, but the Qiskit notebook must be run to confirm its selected settings and test results. The quantum grid was narrowed using earlier results, so matching the number of configurations does not make the tuning histories completely equal.

## Evaluation setup

Both approaches share the same records, split, training-only preprocessing, feature groups, PCA dimensions, balanced SVM class weighting and validation-F1 selection rule. The code uses validation records to choose settings and the separate test records to calculate the reported performance.

Our comparison measures the complete fusion approaches. The classical model uses equal group weights, while the quantum model selects group weights alongside its circuit settings. Both current grids have 24 configurations, but the quantum grid was narrowed using the earlier winning settings. The previously reported gain comes from the custom-statevector experiment; the updated Qiskit implementation needs its own test run.

The earlier results come from the stated stratified split and exact simulated kernels. Repeating the study across data splits and on independent data would help assess how consistently the improvement carries over.

## Quantum resources and runtime

| Resource | Qiskit quantum implementation |
|---|---|
| Qubits per feature-group circuit | 4, reusable across groups |
| Statevector per record and group | 16 complex amplitudes |
| State-preparation operations | 4 Hadamard, 4 RZ and 3 RZZ |
| Training kernel shape | 398 × 398, also used by the classical approach |
| Test-to-training kernel shape | 114 × 398, also used by the classical approach |
| Execution | Exact Qiskit statevector simulation on a classical computer |
| Hardware shots | Not used |
| Configurations tested | 24 quantum and 24 classical |

These operation counts describe one feature-map circuit. We do not use a real quantum processor or estimate fidelity from hardware shots. We have not measured the updated Qiskit notebook's runtime yet, so we do not reuse the original NumPy timing results here.

## Run our project

1. Open [Google Colab](https://colab.research.google.com/) and upload `QSVM_Qiskit_24_Quantum_24_Classical.ipynb`.
2. Upload `Breast_cancer_dataset.csv` to the Colab session. Keep the filename exactly as written, including `.csv`.
3. Run the cells in order. The notebook includes a setup cell with `%pip install -q scikit-learn qiskit` to install the required quantum package.
4. Restart the runtime if installation requests it, then run the notebook again. A CPU runtime is enough for statevector simulation.
5. The notebook prints the number of configurations, selects settings using validation F1, and displays the final test results for both models.

The dataset-loading line is `pd.read_csv("Breast_cancer_dataset.csv")`. The updated notebook uses Qiskit for circuit construction and statevectors; it does not submit jobs to IBM quantum hardware.

## Files in this repository

| File | Contents |
|---|---|
| `QSVM_Qiskit_24_Quantum_24_Classical.ipynb` | Current notebook: data preparation, classical 24-configuration search, Qiskit quantum 24-configuration search and evaluation |
| `Breast_cancer_dataset.csv` | Our 569-record input dataset (use this exact filename when running the notebook) |
| `README.md` | Project explanation, methods, previous results and run instructions |

The notebook also saves processed data and label CSVs for each split, plus `cancer_imputer_scaler.pkl`. These files are generated when the notebook runs. If an older `results.csv` is kept in the repository, its scores describe the earlier run and should not be presented as newly measured Qiskit results.

## Sources and credit

- **Dataset:** [Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Credit to the dataset's creators, Wolberg, Mangasarian and Street. The dataset is listed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Classifiers and preprocessing:** scikit-learn's [SVC](https://scikit-learn.org/1.8/modules/generated/sklearn.svm.SVC.html), [PCA](https://scikit-learn.org/1.8/modules/generated/sklearn.decomposition.PCA.html) and [RBF kernel](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html) implementations.
- **Quantum circuits and statevectors:** [IBM Qiskit](https://qiskit.org/).
- **Quantum-kernel background:** Havlíček et al., [Supervised learning with quantum enhanced feature spaces](https://arxiv.org/abs/1804.11326), for background on quantum feature spaces and kernel methods.

We use these established methods to build and evaluate our feature-group fusion approach, with Qiskit constructing and simulating the quantum states and NumPy calculating the fidelity kernels.
