# SciArx Roadmap

**SciArx** is a scientific computation library written in [ArxLang](https://arxlang.org/) — a
statically-typed, LLVM-powered language with Python-like syntax and native
[Apache Arrow](https://arrow.apache.org/) datatype support. The goal of SciArx is to provide a
comprehensive, high-performance toolkit for scientific and numerical computing in the ArxLang
ecosystem.

> **Note:** SciArx is intentionally phased to follow the maturity of ArxLang itself. Modules
> that require features not yet available in the language (e.g., complex data types, classes,
> generics) are marked accordingly. Where appropriate, SciArx may call into battle-tested C or
> Fortran implementations (BLAS, LAPACK, FFTW, …) via `extern` declarations, or provide its own
> pure-Arx implementations.

---

## Table of Contents

1. [Guiding Principles](#1-guiding-principles)
2. [Phase 0 — Language Prerequisites](#2-phase-0--language-prerequisites)
3. [Phase 1 — Foundation](#3-phase-1--foundation)
4. [Phase 2 — Linear Algebra](#4-phase-2--linear-algebra)
5. [Phase 3 — Signal & Spectral Analysis](#5-phase-3--signal--spectral-analysis)
6. [Phase 4 — Statistics & Probability](#6-phase-4--statistics--probability)
7. [Phase 5 — Optimization](#7-phase-5--optimization)
8. [Phase 6 — Integration & ODEs](#8-phase-6--integration--odes)
9. [Phase 7 — Interpolation & Approximation](#9-phase-7--interpolation--approximation)
10. [Phase 8 — Special Functions](#10-phase-8--special-functions)
11. [Phase 9 — Spatial & Geometry](#11-phase-9--spatial--geometry)
12. [Phase 10 — Sparse Computation](#12-phase-10--sparse-computation)
13. [Phase 11 — Graph Algorithms](#13-phase-11--graph-algorithms)
14. [Phase 12 — Image Processing](#14-phase-12--image-processing)
15. [Phase 13 — Clustering & Distance Metrics](#15-phase-13--clustering--distance-metrics)
16. [Phase 14 — I/O & Interoperability](#16-phase-14--io--interoperability)
17. [Phase 15 — Parallel & Distributed Computing](#17-phase-15--parallel--distributed-computing)
18. [Versioning Strategy](#18-versioning-strategy)

---

## 1. Guiding Principles

- **ArxLang-first.** Every public API is written in `.arx` source. Low-level heavy lifting may be
  delegated to `extern` C/Fortran routines (BLAS, LAPACK, FFTW, etc.) or to purpose-built
  Arx implementations, decided on a case-by-case basis.
- **Arrow-native data model.** Arrays, tensors, and tables align with Apache Arrow memory
  layouts so that data can be shared with other tools at zero-copy cost.
- **Phased delivery.** SciArx releases are gated on ArxLang milestone readiness (type system,
  packaging, classes). Each phase calls out the minimum language features it depends on.
- **No mystery.** Every algorithm references its mathematical definition or well-known
  implementation so that contributors can audit correctness.
- **Performance by default.** Generated LLVM IR is the primary performance lever; optional
  backend bindings provide verified, audited alternatives.

---

## 2. Phase 0 — Language Prerequisites

SciArx tracks the ArxLang roadmap. The items below must land in ArxLang before the
corresponding SciArx work begins.

| ArxLang Feature                     | Required by SciArx                   |
| ----------------------------------- | ------------------------------------ |
| Static typing & type annotations    | All modules                          |
| `int8/16/32/64`, `float16/64` types | All modules                          |
| `string` type                       | I/O, error messages                  |
| `import` / packaging system         | Module organisation                  |
| Mutable variables                   | In-place array operations            |
| Classes / struct-like types         | Array, Matrix, Tensor, Complex types |
| `while` loop                        | Iterative solvers                    |
| `for` over collections              | Array iteration                      |
| Docstring syntax                    | Library documentation                |
| Generics / parametric types         | Container types                      |
| Apache Arrow datatypes (native)     | Arrow-native arrays                  |

---

## 3. Phase 1 — Foundation

**Module:** `sciarx.core`

The foundation layer provides the primary data containers that every other module builds on. It
is the highest-priority deliverable.

### 1.1 N-dimensional Array (`NDArray`)

- Fixed-shape, statically-typed dense array backed by Arrow buffers.
- Element-wise arithmetic: `+`, `-`, `*`, `/`, `**`, `%`.
- Reduction operations: `sum`, `prod`, `min`, `max`, `mean`, `std`, `var`.
- Shape manipulation: `reshape`, `flatten`, `squeeze`, `expand_dims`, `transpose`.
- Indexing: integer, slice, boolean mask, fancy (multi-index).
- Broadcasting rules (NumPy-compatible semantics).
- Dtype support: `f32`, `f64`, `i32`, `i64`, `i16`, `i8`, `bool`.
- Copy vs. view semantics.

### 1.2 Array Creation Routines

- `zeros`, `ones`, `full`, `empty` — allocated arrays.
- `arange`, `linspace`, `logspace`, `geomspace` — range arrays.
- `eye`, `identity`, `diag`, `tri`, `tril`, `triu` — structured matrices.
- `random.rand`, `random.randn`, `random.randint`, `random.seed` — random arrays.
- `from_arrow` / `to_arrow` — Arrow record batch ↔ NDArray conversion.

### 1.3 Mathematical Utility Functions

Elementary functions operating element-wise on arrays:

- **Trigonometric:** `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`, `hypot`.
- **Hyperbolic:** `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`.
- **Exponential / logarithmic:** `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`.
- **Power / rounding:** `sqrt`, `cbrt`, `abs`, `sign`, `ceil`, `floor`, `round`, `clip`.
- **Bitwise (integer arrays):** `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`,
  `right_shift`.
- **Complex number support:** `real`, `imag`, `angle`, `conj`, `abs` (modulus).

### 1.4 Sorting & Searching

- `sort`, `argsort`, `lexsort` (single- and multi-key).
- `partition`, `argpartition`.
- `searchsorted` (binary search on sorted arrays).
- `unique`, `in1d`, `intersect1d`, `union1d`, `setdiff1d`.

### 1.5 String & I/O Helpers _(depends on `string` type)_

- `savetxt` / `loadtxt` — delimited text files.
- `save` / `load` — binary format (Arrow IPC or custom `.sarx`).
- `savez` — multiple arrays to a single archive.

---

## 4. Phase 2 — Linear Algebra

**Module:** `sciarx.linalg`

Provides dense and structured linear algebra. Initial implementations may call BLAS / LAPACK
via `extern`; pure-Arx fallbacks are a stretch goal.

### 2.1 Basic Linear Algebra

- Matrix multiplication: `matmul`, `dot`, `vdot`, `inner`, `outer`, `kron`.
- Matrix norms: `norm` (Frobenius, L1, L2, L∞, nuclear).
- Vector norms: `norm` (L1, L2, Lp, L∞).
- Trace: `trace`.
- Determinant: `det`, `slogdet`.
- Rank: `matrix_rank`.

### 2.2 Decompositions

- **LU decomposition:** `lu` (with partial pivoting, via LAPACK `dgetrf`).
- **QR decomposition:** `qr` (full, reduced, pivoted).
- **Cholesky decomposition:** `cholesky` (for symmetric positive-definite matrices).
- **Singular Value Decomposition (SVD):** `svd` (full, economy, truncated).
- **Eigenvalue problems:**
  - `eig` — general eigenvalues/eigenvectors.
  - `eigh` — symmetric/Hermitian matrices.
  - `eigvals`, `eigvalsh` — eigenvalues only.
- **Schur decomposition:** `schur`.
- **Hessenberg form:** `hessenberg`.

### 2.3 Solvers

- `solve` — square linear system Ax = b.
- `lstsq` — least-squares solution.
- `inv` — matrix inverse.
- `pinv` — Moore–Penrose pseudoinverse.
- Triangular solvers: `solve_triangular`.
- Banded and symmetric positive-definite specialisations.

### 2.4 Matrix Functions

- `matrix_power` — integer powers.
- `expm` — matrix exponential.
- `logm` — matrix logarithm.
- `sqrtm` — matrix square root.
- `funm` — apply arbitrary scalar function to a matrix.

### 2.5 Structured Matrices

- Companion, Hadamard, Hilbert, inverse Hilbert, Leslie, Pascal, Toeplitz, circulant,
  Hankel matrices.

---

## 5. Phase 3 — Signal & Spectral Analysis

**Module:** `sciarx.signal`

### 5.1 Discrete Fourier Transform

- `fft`, `ifft` — 1-D DFT (may delegate to FFTW via `extern`).
- `fft2`, `ifft2` — 2-D DFT.
- `fftn`, `ifftn` — N-D DFT.
- `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn` — real-input variants.
- `fftfreq`, `rfftfreq` — frequency-bin arrays.
- `fftshift`, `ifftshift` — zero-frequency centering.

### 5.2 Discrete Cosine / Sine Transforms

- `dct`, `idct` (types I–IV).
- `dst`, `idst` (types I–IV).

### 5.3 Short-Time Fourier Transform (STFT)

- `stft`, `istft` — windowed spectral analysis and reconstruction.
- Window functions: `hann`, `hamming`, `blackman`, `bartlett`, `flat_top`, `kaiser`,
  `tukey`, `cosine`, `boxcar`, `triang`, `nuttall`, `parzen`.

### 5.4 Convolution & Correlation

- `convolve`, `correlate` — 1-D; `convolve2d`, `correlate2d` — 2-D.
- `fftconvolve` — FFT-based convolution.
- `oaconvolve` — overlap-add method.
- `choose_conv_method` — select optimal algorithm.

### 5.5 Filter Design

- **Analog prototypes:** Butterworth, Chebyshev I/II, Elliptic (Cauer), Bessel.
- **Transformations:** low-pass → low-pass / high-pass / band-pass / band-stop.
- **Digital design:** `bilinear_transform`, `bilinear_zpk`.
- FIR design: `firwin`, `firwin2`, `firls`, `remez` (Parks–McClellan).
- Filter representations: transfer function (`ba`), zero-pole-gain (`zpk`), state-space, SOS.
- Conversions between representations: `tf2zpk`, `zpk2tf`, `tf2sos`, `sos2tf`, `ss2tf`, …

### 5.6 Filter Application

- `lfilter` — direct-form II transposed IIR.
- `sosfilt` — second-order sections (numerically stable).
- `filtfilt` — zero-phase forward-backward filtering.
- `lfiltic` — initial conditions.
- `medfilt`, `medfilt2d` — median filtering.
- `wiener` — Wiener denoising filter.

### 5.7 Spectral Estimation

- `periodogram` — power spectral density.
- `welch` — Welch's method.
- `lombscargle` — Lomb–Scargle for non-uniform data.
- `coherence`, `csd` — coherence and cross-spectral density.

### 5.8 Peak Detection

- `find_peaks` — local maxima with prominence & width constraints.
- `peak_prominences`, `peak_widths`.
- `argrelmin`, `argrelmax`, `argrelextrema`.

### 5.9 Waveform Generation

- `chirp`, `gausspulse`, `square`, `sawtooth`, `unit_impulse`.
- `sweep_poly`.

### 5.10 Resampling

- `resample`, `resample_poly` — polyphase resampling.
- `decimate`, `upfirdn`.

---

## 6. Phase 4 — Statistics & Probability

**Module:** `sciarx.stats`

### 6.1 Descriptive Statistics

- Central tendency: `mean`, `median`, `mode`, `gmean`, `hmean`, `trim_mean`.
- Spread: `std`, `var`, `sem`, `iqr`, `mad` (median absolute deviation).
- Shape: `skew`, `kurtosis`.
- Order statistics: `scoreatpercentile`, `percentileofscore`, `describe`.
- Frequency: `relfreq`, `cumfreq`.

### 6.2 Continuous Probability Distributions

Each distribution exposes `pdf`, `cdf`, `ppf` (quantile), `sf` (survival), `rvs` (random
variates), `fit`, `stats`, `entropy`, `logpdf`, …

- Normal (Gaussian), lognormal, truncated normal.
- Uniform, triangular, beta, Dirichlet.
- Exponential, Laplace, gamma, inverse-gamma, chi-squared, noncentral chi-squared.
- Student-t, noncentral t, F, noncentral F.
- Cauchy, Lévy, Pareto, Weibull (min/max), Gumbel (min/max), logistic.
- Rayleigh, Maxwell, Rice.
- Von Mises, wrapped Cauchy (circular statistics).
- Multivariate normal, multivariate t, Wishart, inverse-Wishart.

### 6.3 Discrete Probability Distributions

- Bernoulli, binomial, negative binomial, hypergeometric, multinomial.
- Poisson, geometric, logarithmic, Zipf, Zipf-Mandelbrot.
- Discrete uniform.

### 6.4 Hypothesis Tests

- **Normality:** Shapiro-Wilk, D'Agostino–Pearson, Kolmogorov–Smirnov (1- & 2-sample),
  Anderson–Darling, Lilliefors.
- **Location (parametric):** one-sample t-test, independent-samples t-test, paired t-test,
  one-way ANOVA, Welch's ANOVA, two-way ANOVA.
- **Location (non-parametric):** Mann–Whitney U, Wilcoxon signed-rank, Kruskal–Wallis,
  Friedman, median test.
- **Association:** Pearson r, Spearman ρ, Kendall τ, point-biserial, chi-squared contingency,
  Fisher exact test, McNemar.
- **Variance:** Bartlett, Levene, Fligner–Killeen.
- **Goodness of fit:** chi-squared, Kolmogorov–Smirnov.
- **Post-hoc / multiple comparisons:** Bonferroni, Holm, Benjamini–Hochberg, Tukey HSD,
  Dunn.

### 6.5 Kernel Density Estimation

- `gaussian_kde` — Gaussian KDE with Silverman / Scott bandwidth.
- Multivariate KDE.

### 6.6 Circular Statistics

- `circmean`, `circstd`, `circvar`, `circmoment`.
- `rayleightest`, `vtest` — uniformity tests.

### 6.7 Sampling & Monte Carlo

- `qmc` — Quasi-Monte Carlo: Halton, Sobol, Latin Hypercube sequences.
- Bootstrap resampling utilities.

---

## 7. Phase 5 — Optimization

**Module:** `sciarx.optimize`

### 7.1 Scalar Minimization

- `minimize_scalar` — Brent, golden-section, bounded.

### 7.2 Unconstrained Multivariate Minimization

- Nelder–Mead simplex.
- Powell's conjugate-direction method.
- Conjugate gradient (CG, Fletcher–Reeves, Polak–Ribière).
- BFGS and L-BFGS-B.
- Newton-CG.
- Trust-region methods: `dogleg`, `trust-ncg`, `trust-krylov`, `trust-exact`.

### 7.3 Constrained Minimization

- SLSQP (Sequential Least Squares Programming).
- COBYLA (Constrained Optimization BY Linear Approximations).
- `LinearConstraint`, `NonlinearConstraint`, `Bounds` objects.

### 7.4 Global Optimization

- `differential_evolution` — stochastic evolutionary algorithm.
- `basinhopping` — stochastic basin-hopping.
- `dual_annealing` — simulated annealing with dual-phase strategy.
- `shgo` — simplicial homology global optimization.
- `direct` — DIRECT algorithm for Lipschitz functions.

### 7.5 Root Finding

- `brentq`, `brenth`, `bisect`, `ridder`, `toms748` — bracketed scalar roots.
- `newton`, `secant`, `halley` — open methods.
- `fsolve` — multivariate root finding (Newton-Krylov wrapper).
- `root` — unified interface: `hybr`, `lm`, `broyden1`, `broyden2`, `anderson`,
  `krylov`, `df-sane`.

### 7.6 Linear Programming & Mixed-Integer

- `linprog` — simplex, revised simplex, interior point (HiGHS backend).
- `milp` — mixed-integer linear programming.

### 7.7 Least Squares

- `least_squares` — nonlinear (LM, Trust Region Reflective, dogbox); bounds support.
- `lsq_linear` — linear bounded least squares.
- `curve_fit` — non-linear curve fitting.
- `nnls` — non-negative least squares.
- `lstsq` — unconstrained linear least squares.

### 7.8 Assignment Problems

- `linear_sum_assignment` — Hungarian algorithm.
- `quadratic_assignment` — FAQ and 2-opt heuristics.

---

## 8. Phase 6 — Integration & ODEs

**Module:** `sciarx.integrate`

### 8.1 Numerical Quadrature (Definite Integrals)

- `quad` — adaptive Gauss–Kronrod (QUADPACK).
- `dblquad`, `tplquad` — double and triple integrals.
- `nquad` — N-dimensional adaptive quadrature.
- `fixed_quad` — fixed-order Gaussian quadrature.
- `quadrature` — adaptive Gaussian quadrature.
- `romberg` — Romberg integration.
- `newton_cotes` — Newton–Cotes weights.
- `simpson`, `trapezoid`, `cumulative_trapezoid` — simple rules on sampled data.

### 8.2 Ordinary Differential Equations (IVP)

- `solve_ivp` — unified interface with solvers:
  - `RK23` — explicit Runge–Kutta order 3(2).
  - `RK45` — explicit Runge–Kutta order 5(4) (Dormand–Prince).
  - `DOP853` — explicit Runge–Kutta order 8.
  - `Radau` — implicit Runge–Kutta (for stiff problems).
  - `BDF` — implicit multi-step (LSODA-style).
  - `LSODA` — automatic stiffness detection.
- Dense output (continuous solution interpolation).
- Event detection with termination support.

### 8.3 Boundary Value Problems (BVP)

- `solve_bvp` — collocation method for BVPs of the form y' = f(t, y).

---

## 9. Phase 7 — Interpolation & Approximation

**Module:** `sciarx.interpolate`

### 9.1 1-D Interpolation

- `interp1d` — linear, nearest, zero, slinear, quadratic, cubic.
- `CubicSpline` — natural / clamped / not-a-knot / periodic boundary conditions.
- `Akima1DInterpolator` — Akima spline.
- `PchipInterpolator` — PCHIP monotone cubic spline.
- `BarycentricInterpolator` — barycentric Lagrange.
- `KroghInterpolator` — Hermite interpolation.

### 9.2 Splines

- `BSpline` — B-spline evaluation, derivatives, antiderivatives.
- `make_interp_spline` — data-fitted B-spline.
- `make_lsq_spline` — least-squares B-spline.
- `UnivariateSpline`, `InterpolatedUnivariateSpline`, `LSQUnivariateSpline`.

### 9.3 Multivariate Interpolation

- `RegularGridInterpolator` — rectilinear grids (linear, nearest, cubic, quintic).
- `interpn` — convenience wrapper.
- `NearestNDInterpolator` — nearest-neighbour for scattered data.
- `LinearNDInterpolator` — barycentric linear interpolation (Delaunay).
- `CloughTocher2DInterpolator` — C1 cubic interpolation (scattered 2-D).
- `RBFInterpolator` — radial basis function (thin-plate, multiquadric, Gaussian, …).

### 9.4 Polynomial Approximation

- Polynomial class with arithmetic, roots, evaluation, integration, differentiation.
- Chebyshev, Legendre, Laguerre, Hermite polynomial series.
- Padé approximation.

---

## 10. Phase 8 — Special Functions

**Module:** `sciarx.special`

All functions support scalar and array inputs.

### 10.1 Airy & Related

- `airy`, `airye` — Airy functions Ai, Bi and their derivatives.
- `ai_zeros`, `bi_zeros` — zeros of Airy functions.

### 10.2 Bessel Functions

- `jv`, `yv`, `iv`, `kv` — Bessel J, Y, I, K of real order.
- `jn_zeros`, `yn_zeros`, `jnp_zeros`, `ynp_zeros` — zeros.
- `spherical_jn`, `spherical_yn`, `spherical_in`, `spherical_kn`.
- Hankel functions `hankel1`, `hankel2`.

### 10.3 Gamma & Related

- `gamma`, `gammaln`, `loggamma`, `gammasgn`.
- `digamma` (`psi`), `polygamma`.
- `gammainc`, `gammaincc`, `gammaincinv`, `gammainccinv` — regularised incomplete gamma.
- `beta`, `betaln`, `betainc`, `betaincinv`.
- `factorial`, `comb`, `perm`.

### 10.4 Error Functions

- `erf`, `erfc`, `erfinv`, `erfcinv`.
- `erfi`, `erfcx` — imaginary and scaled complementary error functions.
- Faddeeva: `wofz`.
- `ndtri` — inverse of normal CDF.

### 10.5 Elliptic Integrals & Functions

- `ellipk`, `ellipkm1` — complete elliptic integral of the first kind.
- `ellipe` — complete elliptic integral of the second kind.
- `ellipj` — Jacobi elliptic functions (sn, cn, dn, ph).
- `elliprd`, `elliprf`, `elliprg`, `elliprj` — Carlson symmetric forms.

### 10.6 Hypergeometric Functions

- `hyp1f1` — confluent hypergeometric (Kummer).
- `hyperu` — Tricomi.
- `hyp2f1` — Gauss hypergeometric.
- `hyp0f1`.

### 10.7 Orthogonal Polynomials

- Legendre, associated Legendre, spherical harmonics.
- Chebyshev T & U, Jacobi, Gegenbauer.
- Hermite (probabilist's and physicist's), Laguerre, associated Laguerre.

### 10.8 Zeta & Number-Theoretic Functions

- `zeta`, `zetac` — Riemann zeta.
- `bernoulli`, `euler` — Bernoulli and Euler numbers.
- `poch` — Pochhammer symbol.

### 10.9 Information-Theoretic

- `entr`, `rel_entr`, `kl_div`, `huber`, `pseudo_huber`.
- `xlogy`, `xlog1py` — stable x·log(y) computation.

### 10.10 Statistical Distribution Functions

- Normal: `ndtr`, `ndtri`, `log_ndtr`.
- Student-t, chi-squared, F, beta, gamma CDFs and inverse CDFs.

---

## 11. Phase 9 — Spatial & Geometry

**Module:** `sciarx.spatial`

### 11.1 k-d Tree & Ball Tree

- `KDTree`, `cKDTree` — nearest-neighbour queries, range queries.
- `BallTree` — metric-space variant.

### 11.2 Convex Hull & Triangulation

- `ConvexHull` — Quickhull algorithm (wraps Qhull).
- `Delaunay` — Delaunay triangulation.
- `Voronoi` — Voronoi diagram.

### 11.3 Distance Computations

- `distance_matrix` — pairwise distance matrix.
- `cdist` — condensed pairwise distances.
- `pdist` — all-pairs distances.
- Metrics: Euclidean, Minkowski, Manhattan (cityblock), Chebyshev, cosine, correlation,
  Hamming, Jaccard, Dice, Russelrao, Canberra, Mahalanobis, seuclidean, braycurtis.

### 11.4 Geometric Transformations

- Rotation matrices: `Rotation` class (from Euler angles, quaternions, axis-angle,
  rotation vectors, DCM).
- Rigid-body transforms: `RigidTransform`.
- `geometric_transform` — general coordinate mapping.

### 11.5 Spherical Geometry

- Great-circle distance.
- Spherical Voronoi: `SphericalVoronoi`.

---

## 12. Phase 10 — Sparse Computation

**Module:** `sciarx.sparse`

### 12.1 Sparse Matrix Formats

- CSR (Compressed Sparse Row).
- CSC (Compressed Sparse Column).
- COO (Coordinate format — for construction).
- BSR (Block Sparse Row).
- LIL (List of Lists — for incremental construction).
- DOK (Dictionary of Keys).
- Conversion between formats.

### 12.2 Sparse Linear Algebra

**Module:** `sciarx.sparse.linalg`

- `spsolve` — direct solver (wraps SuiteSparse / UMFPACK via `extern`).
- `spsolve_triangular`, `splu`, `spilu` — LU decompositions.
- **Iterative solvers:** `cg`, `cgs`, `bicg`, `bicgstab`, `gmres`, `lgmres`, `minres`,
  `qmr`, `gcrotmk`, `tfqmr`.
- **Preconditioners:** `LinearOperator`, `spilu` as preconditioner.
- **Eigenvalue solvers:** `eigs` (ARPACK), `eigsh`, `svds`.
- `norm` — sparse matrix norms.
- `expm` — sparse matrix exponential.
- `expm_multiply` — action of matrix exponential on a vector.

### 12.3 Sparse Graph Algorithms

**Module:** `sciarx.sparse.csgraph`

- `shortest_path`, `dijkstra`, `bellman_ford`, `johnson` — shortest paths.
- `minimum_spanning_tree` — Kruskal.
- `breadth_first_order`, `depth_first_order`, `breadth_first_tree`, `depth_first_tree`.
- `connected_components`, `laplacian`, `structural_rank`.

---

## 13. Phase 11 — Graph Algorithms

**Module:** `sciarx.graphs`

Higher-level graph algorithms complementing the sparse graph module.

- Graph representation: adjacency list, adjacency matrix, edge list.
- Topological sort, cycle detection.
- Strongly connected components (Tarjan, Kosaraju).
- Betweenness, closeness, degree, eigenvector centrality.
- Community detection (Louvain, label propagation).
- Maximum flow / minimum cut (Edmonds–Karp, Dinic).
- Bipartite matching.
- Spectral graph theory: graph Laplacian, normalised Laplacian, spectral embedding.

---

## 14. Phase 12 — Image Processing

**Module:** `sciarx.ndimage`

Primarily N-dimensional array operations applicable to images and volumetric data.

### 14.1 Filters

- Gaussian, uniform (box), median, percentile, generic, rank filters.
- Laplacian, Sobel, Prewitt, Scharr (gradient-based).
- Bilateral filter.
- `convolve`, `correlate` — generic kernel convolution (N-D).
- Gabor filter.

### 14.2 Morphological Operations

- `binary_erosion`, `binary_dilation`, `binary_opening`, `binary_closing`.
- `grey_erosion`, `grey_dilation`, `grey_opening`, `grey_closing`.
- `binary_fill_holes`, `binary_propagation`.
- `distance_transform_edt`, `distance_transform_bf`, `distance_transform_cdt`.

### 14.3 Measurements

- `label`, `find_objects` — connected-component labelling.
- `center_of_mass`, `moments`, `inertia_tensor`.
- `histogram`, `mean`, `standard_deviation`, `variance`, `minimum`, `maximum`.
- `sum_labels`, `median`, `percentile`.

### 14.4 Interpolation & Geometric Transforms

- `zoom` — upsampling / downsampling.
- `rotate` — rotation by arbitrary angle.
- `shift`, `affine_transform`, `map_coordinates`.
- `geometric_transform` — user-defined coordinate mapping.

### 14.5 Fourier-Domain Filtering

- `fourier_gaussian`, `fourier_uniform`, `fourier_ellipsoid`, `fourier_shift`.

---

## 15. Phase 13 — Clustering & Distance Metrics

**Module:** `sciarx.cluster`

### 15.1 Hierarchical Clustering

- `linkage` — single, complete, average (UPGMA), weighted (WPGMA), Ward, centroid, median.
- `dendrogram` — dendrogram computation (not rendering).
- `fcluster`, `fclusterdata` — flat cluster extraction.
- `cut_tree` — cut dendrogram at given levels.
- `inconsistent`, `cophenet` — validity measures.

### 15.2 Vector Quantisation (k-Means & Variants)

- `kmeans`, `kmeans2` — Lloyd's algorithm.
- `vq` — vector quantisation.
- Mini-batch k-Means.
- k-Medoids.

### 15.3 Cluster Validity

- Silhouette score.
- Davies–Bouldin index.
- Calinski–Harabasz index.

---

## 16. Phase 14 — I/O & Interoperability

**Module:** `sciarx.io`

### 16.1 File Formats

- Plain text / CSV / TSV — `read_csv`, `write_csv`.
- Apache Arrow IPC (stream & file) — `read_ipc`, `write_ipc`.
- Parquet — `read_parquet`, `write_parquet` (via Arrow).
- HDF5 — `read_hdf5`, `write_hdf5` (via `extern` C HDF5 API).
- MATLAB `.mat` files (v5 and v7.3) — `loadmat`, `savemat`.
- Fortran unformatted binary — `FortranFile`.
- WAV audio — `read_wav`, `write_wav`.
- NetCDF — `read_netcdf`, `write_netcdf` (via `extern` netCDF-C).
- JSON & MessagePack — `read_json`, `write_json`.

### 16.2 Foreign Function Interfaces

- `extern` declarations for C shared libraries.
- Fortran interoperability via ISO_C_BINDING-compatible `extern`.
- Memory layout helpers (row-major ↔ column-major).

### 16.3 Interoperability Utilities

- Arrow ↔ NDArray zero-copy bridge.
- Type-coercion utilities (`astype`, `broadcast_to`, `contiguous`).

---

## 17. Phase 15 — Parallel & Distributed Computing

**Module:** `sciarx.parallel`

This phase depends on ArxLang gaining concurrency primitives.

- Thread-level parallelism for embarrassingly-parallel array operations.
- Multi-process task dispatch (`pmap` — parallel map over arrays).
- OpenMP-compatible loops via LLVM loop annotations.
- GPU dispatch (CUDA / Metal / SYCL via LLVM offloading) — stretch goal.
- Chunked / out-of-core processing for arrays larger than RAM.
- Distributed linear algebra (MPI-based, via `extern` ScaLAPACK) — stretch goal.

---

## 18. Versioning Strategy

SciArx follows [semantic versioning](https://semver.org/). Release milestones are tightly
coupled to ArxLang language milestones.

| SciArx Version | ArxLang Prerequisite              | Key SciArx Content                                                                          |
| -------------- | --------------------------------- | ------------------------------------------------------------------------------------------- |
| **0.1**        | f32 type, `extern`, functions     | Core NDArray (f32 only), basic math                                                         |
| **0.2**        | Static typing, int/float variants | Full dtype support, linalg basics                                                           |
| **0.3**        | `import`, packaging               | Module organisation, public API                                                             |
| **0.4**        | Classes / structs                 | Proper Array/Matrix types, complex                                                          |
| **0.5**        | Arrow datatypes                   | Arrow-native arrays, I/O                                                                    |
| **0.6**        | Iterative & while loops           | ODE solvers, iterative linear solvers                                                       |
| **1.0**        | Full language stability           | Complete Phase 1–8 (core, linalg, signal, stats, optimize, integrate, interpolate, special) |
| **1.x**        | —                                 | Phase 9–15: spatial, sparse, graphs, ndimage, cluster, I/O, parallel                        |

---

_Last updated: March 2026_
