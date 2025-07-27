# ML From Scratch — Linear/Logistic Regression, SVM & KNN  
**NumPy-only implementations + clean notebooks + step‑by‑step animations**

End‑to‑end mini‑projects that rebuild four core ML algorithms **from first principles**, then apply them to small datasets with clear visualizations:

- **Linear Regression** → salary vs. experience (MSE + gradient descent)  
- **Logistic Regression** → diabetes classification (log‑loss + gradient descent)  
- **Linear SVM (soft margin)** → hinge loss + L2 regularization (SGD)  
- **k‑Nearest Neighbors (KNN)** → Euclidean/Manhattan distance + majority vote  

Animated GIFs show training dynamics and decision boundaries in 2‑D.

---

## Table of Contents
1. [Project Motivation](#motivation)  
2. [What’s Implemented](#implemented)  
3. [Datasets](#datasets)  
4. [Model Summaries & Results](#summaries)  
5. [Animations](#animations)  
6. [Limitations & Next Steps](#next)  
7. [Author](#author)

---

<a name="motivation"></a>
## 1) Project Motivation
I wanted a **hands‑on, inspection‑friendly** view of classic ML. Instead of relying on libraries for the core math, I re‑implemented the learning rules in **NumPy** and kept the code compact and well‑commented. The goal is to:
- demystify the **objective functions** and **updates**,
- show **how scaling and hyperparameters** affect behavior,
- and make learning **visible** via simple **matplotlib** animations.

---

<a name="implemented"></a>
## 2) What’s Implemented
- **Linear Regression (from scratch)**  
  - Hypothesis $\hat y = w^\top x + b$
  - Loss: **Mean Squared Error (MSE)**  
  - Optimizer: **Batch Gradient Descent**  
- **Logistic Regression (from scratch)**  
  - Hypothesis $\hat y=\sigma(w^\top x + b)$  
  - Loss: **Binary Cross‑Entropy / Log‑loss**  
  - Optimizer: **Batch Gradient Descent**  
- **Support Vector Machine — Linear, Soft‑Margin (from scratch)**  
  - Loss: **Hinge** $\max(0, 1 - y(w^\top x - b))$ with **L2** penalty  
  - Optimizer: **Stochastic Gradient Descent** style updates  
- **k‑Nearest Neighbors (from scratch)**  
  - Distances: **Euclidean** / **Manhattan**  
  - **Majority vote** over top‑k neighbors (odd $k$ suggested for binary data)

All four are implemented without scikit‑learn (sklearn is used only for **data split**, **scaling**, and, for SVM/KNN demos, **baseline comparisons/visualization** where appropriate).

---

<a name="datasets"></a>
## 3) Datasets
- **Salary** (toy regression): `salary_data.csv`  
  *1 feature (years of experience) → target (salary)*  
- **Diabetes** (binary classification): `diabetes2.csv`  
  *8 numeric features → target (Outcome ∈ {0,1})*  
  - **Standardization** (`StandardScaler`) is applied before Logistic/SVM/KNN.

> For 2‑D boundary animations I pick two informative features (**Glucose** & **BMI**) so the decision surface can be plotted.

---

## 4) Model Summaries & Results
### Linear Regression (Salary)
  - **Objective:** minimize MSE; updates via gradient descent.
  - **Behavior:** line converges quickly with a small learning rate; loss curve decreases smoothly.
  - **Animation:** shows the line fitting data + loss decreasing per iteration.

### Logistic Regression (Diabetes)
- **Objective:** Log‑loss; probabilities via **sigmoid**; 0.5 threshold → class.  
- **Preprocessing:** **Standardize** all 8 features (`StandardScaler`).  
- **Result:** Competitive train/test accuracy with appropriate learning rate/iterations.  
- **Animation:** Live **cost curve** + **sigmoid curve** while varying one feature (others fixed at mean).

### Linear SVM (Soft Margin)
- **Objective:** **Hinge loss** + **L2** penalty; SGD‑style updates with regularization **λ**.  
- **Intuition:** Balance **wide margin** vs. **violations** (controlled by **C/λ**).  
- **Result:** Stable linear separator after scaling.  
- **Animation:** Baseline sklearn SVM varying **C** to visualize the fit–margin trade‑off.

### k‑Nearest Neighbors
- **Procedure:** For a test point, compute **distance to every training point**, sort, take **top‑k**, **majority vote**.  
- **Metrics:** Euclidean or Manhattan; use **odd k** for binary classes to avoid ties.  
- **Complexity:** Naive prediction is **O(n log n)** due to sorting—fine for small demos.  
- **Animation:** Decision regions change as **k** grows (jagged for small **k**, smoother for large **k**).


<a name="animations"></a>

## 5) Animations

- `assets/linear_regression.gif` — MSE curve + line fitting + toy gradient‑descent path.  
- `assets/logistic_training.gif` — log‑loss over iterations + sigmoid decision curve.  
- `assets/svm_c.gif` — how **C** tightens/loosens the SVM margin.  
- `assets/knn_k.gif` — how KNN boundaries evolve with **k**.

> For boundary plots I intentionally use **two features** (e.g., *Glucose* & *BMI*) so the surface can be visualized in 2‑D.


<a name="next"></a>

## 6) Limitations & Next Steps
**Limitations**

- From‑scratch focus: no advanced optimizers or non‑linear SVM kernels yet.  
- Efficiency: KNN uses a naive search (no KD‑tree/ball‑tree).  
- Probability calibration only for Logistic; others could add calibration.

**Next Steps**

- Add unit tests for gradient signs and loss monotonicity.  
- Extend SVM to **RBF/Polynomial** kernels and add simple cross‑validation helpers.  
- Implement faster neighbor search for KNN and optional probability calibration.

<a name="author"></a>

## 7) Author

**Laylo Karimova** — From‑scratch ML implementations with clear visuals and concise explanations.

**Topics:** `machine-learning`, `numpy`, `from-scratch`, `linear-regression`, `logistic-regression`, `svm`, `knn`, `jupyter-notebook`, `matplotlib`, `animation`
