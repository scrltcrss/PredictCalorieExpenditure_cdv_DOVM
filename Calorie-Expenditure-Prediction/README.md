# Calorie Expenditure Prediction - Neural Network Architecture Comparison

Advanced neural network solution for predicting calorie expenditure during workouts using PyTorch. This project implements multiple architectures with comprehensive evaluation using **MAPE (Mean Absolute Percentage Error)** as the primary metric.

## Project Overview

This project tackles calorie expenditure prediction with:

- **Multiple Neural Network Architectures**: SimpleNet, DeepNet, WideNet, ResidualNet, and AdaptiveNet
- **Comprehensive Data Analysis**: EDA with visualizations and statistical analysis
- **Advanced Training Pipeline**: KFold cross-validation, early stopping, learning rate scheduling
- **Dropout Comparison**: Models trained with and without dropout for regularization analysis
- **Hyperparameter Optimization**: Systematic comparison of learning rates, batch sizes, and architectures
- **Primary Metric**: MAPE (Mean Absolute Percentage Error) for accurate percentage-based error measurement

---

## Konfiguracja

1. **Zainstaluj Git**
   Instrukcja instalacji: [https://github.com/git-guides/install-git](https://github.com/git-guides/install-git)

2. **Zainstaluj `uv`**
   `uv` to narzędzie do zarządzania środowiskiem i uruchamiania skryptów w projekcie.

```bash
pip install uv
```

Instrukcja instalacji: [https://docs.astral.sh/uv/getting-started/installation/#standalone-installer](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

---

## Inicjalizacja projektu

1. **Sklonuj repozytorium**

```bash
git clone <url>
```

2. **Utwórz nowy branch**

```bash
git checkout -b <nazwa_brancha>
```

3. **Stwórz środowisko wirtualne i zsynchronizuj zależności**

```bash
uv sync
```

---

## Usage

### 1. Data Analysis

Run comprehensive exploratory data analysis:

```bash
uv run python src/run_analysis.py
```

Generates visualizations and statistics in `analysis_results/`:
- Feature distributions
- Correlation matrix
- Gender-based analysis
- Target distribution analysis

### 2. Model Training

#### Default Comprehensive Comparison

```bash
uv run python src/run_training.py --mode default
```

Trains all architectures with dropout variations.

#### Quick Testing

```bash
uv run python src/run_training.py --mode quick --epochs 50 --folds 3
```

#### Custom Configuration

```bash
uv run python src/run_training.py --mode custom --epochs 100 --folds 5
```

### 3. Generate Predictions

#### Use Best Model Automatically

```bash
uv run python src/run_prediction.py
```

#### Use Specific Experiment

```bash
uv run python src/run_prediction.py \
    --experiment-name simple_nodropout_lr0.001 \
    --architecture simple \
    --dropout 0.0
```

### 4. Run Unit Tests

```bash
uv run python -m pytest
```

---

## Project Structure

```
Calorie-Expenditure-Prediction/
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   └── sample_submission.csv
├── src/
│   ├── architectures.py       # Neural network architectures (SimpleNet, DeepNet, WideNet, ResidualNet, AdaptiveNet)
│   ├── kfold_trainer.py       # KFold cross-validation training with early stopping
│   ├── model_comparison.py    # Model comparison framework
│   ├── data_analysis.py       # EDA and visualization utilities
│   ├── predict.py             # Prediction and submission generation
│   ├── run_analysis.py        # Main script for data analysis
│   ├── run_training.py        # Main script for model training
│   └── run_prediction.py      # Main script for predictions
├── experiments/               # Training results
│   ├── comparison_summary.json
│   └── {experiment_name}/
├── analysis_results/          # EDA results 
├── config/
│   └── config.yaml
└── pyproject.toml
```

---

## Neural Network Architectures

1. **SimpleNet** (Baseline)
   - 3 hidden layers: 32 → 64 → 32
   - BatchNormalization + SiLU activation
   - Optional dropout

2. **DeepNet**
   - 5 hidden layers: 64 → 128 → 256 → 128 → 64
   - Increased capacity for complex patterns

3. **WideNet**
   - 3 wide layers: 128 → 256 → 128
   - Fewer layers but larger width

4. **ResidualNet**
   - Residual blocks with skip connections
   - Better gradient flow for deeper networks

5. **AdaptiveNet**
   - Attention-like feature weighting mechanism
   - Dynamic feature importance learning

---

## Training Features

- **KFold Cross-Validation**: 5-fold CV for robust evaluation
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Log Transform**: Uses log1p transformation for target variable
- **Normalization**: Feature standardization using training statistics
- **Multiple Metrics**: **MAPE (primary)**, RMSE, MAE, R²

---

## Evaluation Metrics

### Primary Metric: MAPE (Mean Absolute Percentage Error)

MAPE is the key metric used for model comparison:

```
MAPE = (1/n) * Σ|((actual - predicted) / actual)| * 100%
```

**Advantages:**
- Intuitive percentage-based interpretation
- Scale-independent comparison
- Directly measures prediction accuracy

**Additional Metrics:**
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

---

## Model Comparison Methodology

1. **Architecture Comparison**: Test all 5 architectures
2. **Dropout Analysis**: Compare models with dropout (0.2) vs without (0.0)
3. **Hyperparameter Tuning**: 
   - Learning rates: [0.0005, 0.001, 0.002]
   - Batch sizes: [128, 256, 512]
4. **Ensemble Prediction**: Average predictions from all folds

---

## Results Interpretation

After training, results are saved in `experiments/`:

```
experiments/
├── comparison_summary.json          # Overall comparison (sorted by MAPE)
├── comparison_report.png            # Visual comparison charts
└── {experiment_name}/
    ├── model_fold_*.pth             # Trained model checkpoints
    ├── oof_predictions.npy          # Out-of-fold predictions
    ├── test_predictions.npy         # Test set predictions
    └── experiment_results.json      # Detailed results
```

---

## Key Implementation Details

### Log Transformation

The target variable (Calories) is log-transformed during training:

```python
y_train = torch.log1p(y_train)  # Training
predictions = torch.expm1(predictions)  # Inference
```

### Feature Normalization

Features are normalized using training set statistics:

```python
X_normalized = (X - X_train.mean()) / X_train.std()
```

### Ensemble Prediction

Final predictions are averaged across all K folds:

```python
final_prediction = mean([fold_1_pred, fold_2_pred, ..., fold_k_pred])
```

**Note:** Clipping of predictions to training data range is disabled by default to allow natural model predictions.

---

## Dependencies

- Python ≥ 3.11
- PyTorch ≥ 2.9.0
- pandas ≥ 2.3.3
- numpy
- scikit-learn
- matplotlib ≥ 3.10.7
- seaborn
- hydra-core ≥ 1.3.2

---

## Competition Submission

The generated `submission.csv` is ready for Kaggle submission:

```csv
id,Calories
0,150.234
1,34.567
...
```

---

## Performance Tips

1. **GPU Acceleration**: Automatically uses CUDA if available
2. **Batch Size**: Larger batches (256-512) work well for this dataset
3. **Early Stopping**: Patience of 10 epochs prevents overfitting
4. **Learning Rate**: 0.001 is a good starting point
5. **Folds**: 5 folds provide good bias-variance tradeoff

---

## License

This project is for educational purposes.

---

## Treść zadania programistycznego

Konkurs: [Kaggle Playground Series S5E5](https://www.kaggle.com/competitions/playground-series-s5e5/overview)

Waszym zadaniem jest dobranie odpowiedniej **architektury sieci neuronowej** do predykcji liczby spalanych kalorii podczas treningu.

Projekt powinien obejmować:

* Podział danych na treningowe i walidacyjne w celu ewaluacji jego jakości i uniknięcia przeuczenia
* Porównanie różnych architektur sieci i hiperparametrów (np. `learning rate`, `momentum`, `batch size`)
* Porównanie jakości modeli **z użyciem i bez użycia dropout**
* Skrypt do predykcji, który zwraca wyniki w formacie określonym na stronie konkursu
* Wizualizacje danych i część analityczną, pokazującą interesujące zależności w danych

Finalny rezultat:

* Utworzenie repozytorium na platformie GitHub z rozwiązaniem zadania oraz uworzeniem Merge Request'a do głównego branch'a (kod powinien znajdować się na branchu developerskim)
* Dodanie konta `jfraszczakcdv` jako contributor'a repozytorium
* Zamieszczenie w repozytorium drobnego raportu obejmujacego analizę danych
* Zamieszczenie swoich predykcji na danych testowych na stronie konkursu


## Treść zadania teoretycznego

1. Oblicz pochodne sieci neuronowej o następującej architekturze:

* Wejście: 2 cechy -> [x1, x2]  
* Pierwsza warstwa ukryta: 2 neurony + ReLU 
* Druga warstwa ukryta: 1 neuron + ReLU
* Wyjście: 1 neuron -> y
* Funkcja straty: Mean Squared Error
* Jedna obserwacja wejściowa [x1, x2] -> [2, 3], y -> 5

2. Odpowiedz na pytania:

* Dla jakich rodzajów zadań warto rozpatrzyć użycie sieci neuronowej. Dlaczego nie można napisać ręcznie programu do predykcji wartości?
* W jakim celu używane są funkcje aktywacji? Co się stanie jeśli w sieci o wielu warstwach ukrytych pozbędziemy się funkcji aktywacji?
* Wyjaśnij rolę dropout'u w trenowaniu sieci neuronowych.
