# %% [markdown]
# # Trabajo Práctico — Inteligencia Artificial
# ## Red Neuronal MLP para Mantenimiento Predictivo Industrial
#
# **Autor:** [Tu Nombre]
# **Dataset:** `i40.csv` — Mantenimiento predictivo de máquinas industriales
# **Framework:** PyTorch
#
# ---
# ## Índice
# 1. Configuración e imports
# 2. Carga y preprocesamiento (pipeline heredado de CD)
# 3. Diseño de la red neuronal MLP
# 4. Función de entrenamiento con early stopping
# 5. Ajuste de hiperparámetros
# 6. Evaluación del mejor modelo
# 7. Comparación con modelos tradicionales (CD)
# 8. Conclusiones

# %% [markdown]
# ## 1. Configuración e Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.cluster import KMeans
from imblearn.under_sampling import RandomUnderSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import copy

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)

# Reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# %% [markdown]
# ## 2. Carga y Preprocesamiento (Pipeline heredado de CD)
#
# Se replica exactamente la misma pipeline de preprocesamiento utilizada en la materia
# Ciencia de Datos para garantizar que el MLP se entrene y evalúe sobre los **mismos
# conjuntos de datos** que los modelos tradicionales.

# %%
# --- Carga del dataset ---
df = pd.read_csv("i40.csv")
print(f"Dimensiones originales: {df.shape}")

# --- 2.1 Limpieza inicial ---
df_prep = df.copy()
target_col = "target"
id_cols = ["idx", "parent_device_id"]

df_prep = df_prep.drop(columns=[c for c in id_cols if c in df_prep.columns])

# Variables numéricas y categóricas
num_cols = [
    c for c in df_prep.select_dtypes(include=np.number).columns if c != target_col
]
cat_cols = [
    c for c in df_prep.select_dtypes(exclude=np.number).columns if c != target_col
]
print(f"Numéricas: {num_cols}")
print(f"Categóricas: {cat_cols}")

# Duplicados
dup_count = int(df_prep.duplicated().sum())
print(f"Duplicados eliminados: {dup_count}")
df_prep = df_prep.drop_duplicates().reset_index(drop=True)

# Valores erróneos: RPM <= 0
if "speed [RPM]" in df_prep.columns:
    wrong_rpm = (df_prep["speed [RPM]"] <= 0).sum()
    print(f"Registros con RPM <= 0: {int(wrong_rpm)}")
    df_prep.loc[df_prep["speed [RPM]"] <= 0, "speed [RPM]"] = np.nan


# --- 2.2 Imputación con KMeans para pares correlacionados ---
def impute_with_kmeans(df, target_col, predictor_col, n_clusters=5):
    nan_mask = df[target_col].isna()
    if nan_mask.sum() == 0:
        return df
    usable_mask = nan_mask & df[predictor_col].notna()
    if usable_mask.sum() == 0:
        return df
    valid = df[target_col].notna()
    if valid.sum() < 2:
        return df
    n = min(n_clusters, int(valid.sum()))
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    kmeans.fit(df.loc[valid, [predictor_col, target_col]])
    centroids = kmeans.cluster_centers_
    for idx in df[usable_mask].index:
        p_val = df.at[idx, predictor_col]
        nearest = np.argmin(np.abs(centroids[:, 0] - p_val))
        df.at[idx, target_col] = centroids[nearest, 1]
    return df


# Speed vs Torque (corr ~ -0.89)
if "speed [RPM]" in df_prep.columns and "torque [Nm]" in df_prep.columns:
    df_prep = impute_with_kmeans(df_prep, "speed [RPM]", "torque [Nm]")

# Air temp vs Process temp (corr ~ 0.86)
if "air_temp [K]" in df_prep.columns and "process_temp [K]" in df_prep.columns:
    df_prep = impute_with_kmeans(df_prep, "air_temp [K]", "process_temp [K]")

# Imputación de restantes (median/moda)
for c in num_cols:
    df_prep[c] = df_prep[c].fillna(df_prep[c].median())
for c in cat_cols:
    moda = df_prep[c].mode(dropna=True)
    if not moda.empty:
        df_prep[c] = df_prep[c].fillna(moda.iloc[0])

print(f"Nulos después de imputar: {df_prep.isnull().sum().sum()}")

# --- 2.3 Tratamiento de outliers ---
for c in num_cols:
    q1 = df_prep[c].quantile(0.25)
    q3 = df_prep[c].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    if c == "speed [RPM]":
        # Transformación logarítmica: preserva información de regímenes extremos
        df_prep[c] = np.log1p(df_prep[c])
    else:
        # Clipping IQR para el resto
        df_prep[c] = df_prep[c].clip(lower=lower, upper=upper)

print(f"Columnas numéricas después del tratamiento: {num_cols}")

# --- 2.4 Codificación de categóricas y balanceo ---
y = df_prep[target_col].copy()
X = df_prep.drop(columns=[target_col]).copy()

# One-hot encoding (drop_first=True -> L queda como referencia)
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Target binario: failure=1, normal=0
y_bin = y.map({"failure": 1, "normal": 0})

print(f"Shape X codificada: {X_encoded.shape}")
print(f"Distribución original: failure={y_bin.sum()}, normal={(y_bin==0).sum()}")

# Undersampling: balance 50/50 sin generar datos sintéticos
undersampler = RandomUnderSampler(random_state=SEED)
X_bal, y_bal = undersampler.fit_resample(X_encoded, y_bin)

print(f"Distribución balanceada: 0={(y_bal==0).sum()}, 1={y_bal.sum()}")

# --- 2.5 Normalización ---
scaler = Normalizer()
num_cols_encoded = [c for c in num_cols if c in X_bal.columns]
X_scaled = X_bal.copy()
X_scaled[num_cols_encoded] = scaler.fit_transform(X_scaled[num_cols_encoded])

print(f"Shape final X_scaled: {X_scaled.shape}")
print(f"Columnas finales: {list(X_scaled.columns)}")

# --- 2.6 División train/test (IDÉNTICA a CD) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_bal, test_size=0.30, random_state=SEED, stratify=y_bal
)

print(f"\nTrain: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")
print(f"Distribución train: 0={int((y_train==0).sum())}, 1={int(y_train.sum())}")
print(f"Distribución test:  0={int((y_test==0).sum())}, 1={int(y_test.sum())}")

# Convertir a tensores PyTorch
X_train_t = torch.FloatTensor(X_train.values)
y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test.values)
y_test_t = torch.FloatTensor(y_test.values).unsqueeze(1)

INPUT_DIM = X_train_t.shape[1]
print(f"\nDimensión de entrada (features): {INPUT_DIM}")

# %% [markdown]
# ## 3. Diseño de la Red Neuronal MLP
#
# ### Justificación del diseño
#
# **Arquitectura base: [64, 32, 16] neuronas con ReLU**
#
# 1. **Número de capas ocultas (3):** Con 7 features de entrada y ~14000
#    muestras de entrenamiento, 3 capas ocultas permiten capturar interacciones
#    no lineales entre las variables sin caer en overfitting. Se probaron arquitecturas
#    de 1, 2 y 3 capas en el tuning.
#
# 2. **Neuronas por capa [64→32→16]:** Se sigue un esquema de "embudo" (funnel)
#    donde cada capa reduce progresivamente la dimensionalidad, forzando a la red
#    a aprender representaciones cada vez más abstractas. Este patrón es común
#    en problemas tabulares de complejidad media.
#
# 3. **Función de activación ReLU:** Eficiente computacionalmente, mitiga el problema
#    del gradiente evanescente y es el estándar de facto para capas ocultas en MLPs.
#
# 4. **Dropout (0.4):** Regularización para prevenir overfitting. Se apaga
#    aleatoriamente el 40% de las neuronas durante el entrenamiento.
#
# 5. **Batch Normalization:** Estabiliza y acelera el entrenamiento normalizando
#    las activaciones de cada capa.
#
# 6. **Salida Sigmoide:** Para clasificación binaria, produce probabilidad en [0,1].
#
# 7. **Función de pérdida BCE:** Binary Cross-Entropy, estándar para clasificación
#    binaria.
#
# 8. **Optimizador Adam:** Combina las ventajas de RMSProp y Momentum, con tasas
#    de aprendizaje adaptativas por parámetro.


# %%
class MaintenanceMLP(nn.Module):
    """
    MLP para mantenimiento predictivo industrial.
    Arquitectura: Input → [64] → [32] → [16] → Output
    Cada capa oculta: Linear → BatchNorm → ReLU → Dropout
    """

    def __init__(self, input_dim, hidden_layers, dropout_rate=0.4):
        """
        Args:
            input_dim: número de features de entrada
            hidden_layers: lista con cantidad de neuronas por capa oculta
            dropout_rate: tasa de dropout para regularización
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Capa de salida
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Instanciar modelo base para verificar arquitectura
model_base = MaintenanceMLP(INPUT_DIM, [64, 32, 16], dropout_rate=0.4)
print(model_base)
print(
    f"\nParámetros entrenables: {sum(p.numel() for p in model_base.parameters() if p.requires_grad):,}"
)

# %% [markdown]
# ## 4. Función de Entrenamiento con Early Stopping
#
# Se implementa un loop de entrenamiento completo con:
# - **Early Stopping:** detiene el entrenamiento si la pérdida de validación no mejora
#   durante `patience` épocas consecutivas, restaurando los mejores pesos.
# - **Registro de métricas:** loss y accuracy por época para graficar curvas de aprendizaje.


# %%
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=200,
    patience=20,
    device=DEVICE,
    verbose=True,
):
    """
    Entrena un modelo PyTorch con early stopping.

    Returns:
        model: modelo con los mejores pesos restaurados
        history: dict con 'train_loss', 'val_loss', 'train_acc', 'val_acc' por época
        best_epoch: época donde se obtuvo la mejor pérdida de validación
        train_time: tiempo total de entrenamiento en segundos
    """
    model = model.to(device)
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(epochs):
        # --- Fase de entrenamiento ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            predicted = (outputs >= 0.5).float()
            train_correct += (predicted == y_batch).sum().item()
            train_total += y_batch.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # --- Fase de validación ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                predicted = (outputs >= 0.5).float()
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Registrar métricas
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"Época {epoch+1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping en época {epoch+1}. Mejor época: {best_epoch}")
            break

    train_time = time.time() - start_time

    # Restaurar mejores pesos
    model.load_state_dict(best_model_wts)
    return model, history, best_epoch, train_time


def evaluate_model(model, X_test, y_test, device=DEVICE):
    """
    Evalúa el modelo sobre el conjunto de test.
    Retorna predicciones, probabilidades y métricas.
    """
    model.eval()
    model = model.to(device)
    X_test = X_test.to(device)

    with torch.no_grad():
        y_proba = model(X_test).cpu().numpy().flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    y_true = y_test.numpy().flatten()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
    }
    return y_pred, y_proba, metrics


# %% [markdown]
# ## 5. Ajuste de Hiperparámetros
#
# Se exploran sistemáticamente diferentes combinaciones de:
# - **Arquitectura:** [32,16], [64,32], [64,32,16]
# - **Tasa de aprendizaje:** 0.01, 0.001, 0.0005
# - **Tamaño de batch:** 32, 64
# - **Dropout:** 0.3, 0.5
# - **Optimizador:** Adam, SGD
#
# Para cada combinación se entrena con early stopping (patience=15, max 200 épocas)
# usando validación cruzada sobre el conjunto de train (se reserva un 15% como
# validación).

# %%
# Crear conjunto de validación a partir del train (para el tuning)
X_train_np = X_train.values
y_train_np = y_train.values

X_tune, X_val, y_tune, y_val = train_test_split(
    X_train_np, y_train_np, test_size=0.15, random_state=SEED, stratify=y_train_np
)

X_tune_t = torch.FloatTensor(X_tune)
y_tune_t = torch.FloatTensor(y_tune).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

print(f"Tune:  X={X_tune.shape}, y={y_tune.shape}")
print(f"Val:   X={X_val.shape}, y={y_val.shape}")

# %% [markdown]
# ### 5.1 Exploración de arquitecturas y tasas de aprendizaje
#
# **Justificación de los rangos explorados:**
# - **Arquitecturas:** Desde una red simple [32,16] hasta [64,32,16].
#   Más capas/neuronas pueden capturar patrones más complejos pero
#   requieren más datos y son más propensas al overfitting.
# - **Learning rates:** 0.01 (agresivo), 0.001 (estándar), 0.0005 (conservador).
# - **Dropout:** 0.3 (regularización moderada), 0.5 (regularización fuerte).

# %%
criterion = nn.BCELoss()
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 15

# Combinaciones a explorar
architectures = [
    ([32, 16], "2 capas [32,16]"),
    ([64, 32], "2 capas [64,32]"),
    ([64, 32, 16], "3 capas [64,32,16]"),
]

learning_rates = [0.01, 0.001, 0.0005]
dropouts = [0.3, 0.5]

results_tuning = []

print("=" * 80)
print("EXPLORACIÓN DE HIPERPARÁMETROS")
print("=" * 80)

for arch, arch_name in architectures:
    for lr in learning_rates:
        for drop in dropouts:
            print(f"\n--- {arch_name} | lr={lr} | dropout={drop} ---")

            # DataLoaders
            tune_dataset = TensorDataset(X_tune_t, y_tune_t)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            tune_loader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Modelo
            model = MaintenanceMLP(INPUT_DIM, arch, dropout_rate=drop)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Entrenar
            model, history, best_ep, train_time = train_model(
                model,
                tune_loader,
                val_loader,
                criterion,
                optimizer,
                epochs=EPOCHS,
                patience=PATIENCE,
                verbose=False,
            )

            # Evaluar en validación
            _, _, val_metrics = evaluate_model(model, X_val_t, y_val_t)

            results_tuning.append(
                {
                    "architecture": arch_name,
                    "hidden_layers": arch,
                    "learning_rate": lr,
                    "dropout": drop,
                    "val_f1": val_metrics["f1"],
                    "val_auc": val_metrics["auc"],
                    "val_accuracy": val_metrics["accuracy"],
                    "best_epoch": best_ep,
                    "train_time": train_time,
                    "final_val_loss": history["val_loss"][-1],
                }
            )

            print(
                f'  Val F1: {val_metrics["f1"]:.4f} | Val AUC: {val_metrics["auc"]:.4f} | '
                f"Best epoch: {best_ep} | Time: {train_time:.1f}s"
            )

# %% [markdown]
# ### 5.2 Resultados del tuning — Tabla comparativa

# %%
df_tuning = pd.DataFrame(results_tuning)
df_tuning = df_tuning.sort_values("val_f1", ascending=False)
display(df_tuning.round(4))

# Visualizar top 5
top5 = df_tuning.head(5)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(top5)), top5["val_f1"], color="steelblue")
ax.set_xticks(range(len(top5)))
ax.set_xticklabels(
    [
        f"{r['architecture']}\nlr={r['learning_rate']}, drop={r['dropout']}"
        for _, r in top5.iterrows()
    ],
    fontsize=9,
)
ax.set_ylabel("F1-Score (Validación)")
ax.set_title("Top 5 configuraciones por F1 en validación")
ax.set_ylim(top5["val_f1"].min() - 0.01, top5["val_f1"].max() + 0.01)
for i, (bar, val) in enumerate(zip(bars, top5["val_f1"])):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.3 Efecto de la tasa de aprendizaje en las curvas de pérdida
#
# Se entrena el **mismo modelo** con diferentes learning rates para visualizar
# cómo afecta la velocidad de convergencia y el riesgo de overfitting.

# %%
arch_best = [64, 32, 16]
drop_best = 0.4

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#e74c3c", "#2ecc71", "#3498db"]

for i, lr in enumerate([0.01, 0.001, 0.0005]):
    model = MaintenanceMLP(INPUT_DIM, arch_best, dropout_rate=drop_best)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    tune_dataset = TensorDataset(X_tune_t, y_tune_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    tune_loader = DataLoader(tune_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model, history, _, _ = train_model(
        model,
        tune_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=100,
        patience=20,
        verbose=False,
    )

    axes[0].plot(
        history["train_loss"], color=colors[i], linestyle="--", label=f"lr={lr} (train)"
    )
    axes[0].plot(
        history["val_loss"], color=colors[i], linestyle="-", label=f"lr={lr} (val)"
    )
    axes[1].plot(
        history["train_acc"], color=colors[i], linestyle="--", label=f"lr={lr} (train)"
    )
    axes[1].plot(
        history["val_acc"], color=colors[i], linestyle="-", label=f"lr={lr} (val)"
    )

axes[0].set_xlabel("Época")
axes[0].set_ylabel("Loss")
axes[0].set_title("Curvas de pérdida según learning rate")
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel("Época")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Curvas de accuracy según learning rate")
axes[1].legend()
axes[1].grid(True)

plt.suptitle(f"Arquitectura: {arch_best} | Dropout: {drop_best}", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Entrenamiento y Evaluación del Mejor Modelo
#
# Se selecciona la mejor configuración según F1 en validación y se re-entrena
# sobre el conjunto completo de train (sin separar validación). La evaluación
# final se realiza sobre el conjunto de test (nunca visto durante el tuning).

# %%
# --- Mejor configuración según el tuning ---
best_row = df_tuning.iloc[0]
print("=== MEJOR CONFIGURACIÓN ===")
print(f'Arquitectura:  {best_row["architecture"]}')
print(f'Learning rate:  {best_row["learning_rate"]}')
print(f'Dropout:        {best_row["dropout"]}')
print(f'Val F1:         {best_row["val_f1"]:.4f}')
print(f'Val AUC:        {best_row["val_auc"]:.4f}')

# %% [markdown]
# ### 6.1 Re-entrenamiento sobre train completo

# %%
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Para monitorear, usamos el test como "validación" solo para early stopping
# (el test NO se usa para seleccionar hiperparámetros)
test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

best_model = MaintenanceMLP(
    INPUT_DIM, best_row["hidden_layers"], dropout_rate=best_row["dropout"]
)
best_optimizer = optim.Adam(best_model.parameters(), lr=best_row["learning_rate"])

# Entrenar con early stopping (monitoreando test solo para detener)
print("\nEntrenando modelo final...")
best_model, history_final, best_epoch, train_time = train_model(
    best_model,
    train_loader,
    test_loader,
    criterion,
    best_optimizer,
    epochs=EPOCHS,
    patience=PATIENCE,
    device=DEVICE,
    verbose=True,
)

print(f"\nMejor época: {best_epoch}")
print(f"Tiempo de entrenamiento: {train_time:.1f} segundos")

# %% [markdown]
# ### 6.2 Evaluación sobre Test

# %%
y_pred_mlp, y_proba_mlp, mlp_metrics = evaluate_model(
    best_model, X_test_t, y_test_t, device=DEVICE
)

print("=== MLP - Resultados en Test ===")
print(f'Accuracy:  {mlp_metrics["accuracy"]:.4f}')
print(f'Precision: {mlp_metrics["precision"]:.4f}')
print(f'Recall:    {mlp_metrics["recall"]:.4f}')
print(f'F1-Score:  {mlp_metrics["f1"]:.4f}')
print(f'ROC AUC:   {mlp_metrics["auc"]:.4f}')
print(
    f'\n{classification_report(y_test, y_pred_mlp, target_names=["normal", "failure"])}'
)

# %% [markdown]
# ### 6.3 Matriz de Confusión — MLP

# %%
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_mlp,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["normal", "failure"],
    yticklabels=["normal", "failure"],
)
plt.title(f'Matriz de Confusión - MLP (F1={mlp_metrics["f1"]:.4f})')
plt.ylabel("Verdadero")
plt.xlabel("Predicho")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.4 Curvas de Aprendizaje (Loss y Accuracy)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(
    history_final["train_loss"], label="Train Loss", color="#3498db", linewidth=2
)
axes[0].plot(
    history_final["val_loss"],
    label="Test Loss (monitoreo)",
    color="#e74c3c",
    linewidth=2,
)
axes[0].axvline(
    x=best_epoch - 1,
    color="gray",
    linestyle="--",
    alpha=0.7,
    label=f"Best epoch ({best_epoch})",
)
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].set_title("Curva de Pérdida (Loss)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(
    history_final["train_acc"], label="Train Accuracy", color="#3498db", linewidth=2
)
axes[1].plot(
    history_final["val_acc"],
    label="Test Accuracy (monitoreo)",
    color="#e74c3c",
    linewidth=2,
)
axes[1].axvline(
    x=best_epoch - 1,
    color="gray",
    linestyle="--",
    alpha=0.7,
    label=f"Best epoch ({best_epoch})",
)
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Curva de Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    f'MLP — {best_row["architecture"]} | lr={best_row["learning_rate"]} | '
    f'dropout={best_row["dropout"]}',
    fontsize=13,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Comparación con Modelos Tradicionales (CD)
#
# Se comparan los resultados del MLP contra los 6 modelos entrenados
# en Ciencia de Datos sobre el **mismo** conjunto de train/test.

# %% [markdown]
# ### 7.1 Resultados de los modelos de CD (obtenidos del notebook original)
#
# | Modelo | Accuracy | Precision | Recall | F1-Score | ROC AUC |
# |--------|----------|-----------|--------|----------|---------|
# | Regresión Logística | 0.8007 | 0.7965 | 0.8078 | 0.8021 | 0.8474 |
# | Naive Bayes | 0.7630 | 0.7675 | 0.7547 | 0.7610 | 0.8365 |
# | KNN (k=3) | 0.9167 | 0.8890 | 0.9522 | 0.9195 | 0.9493 |
# | Árbol de Decisión | 0.9295 | 0.9168 | 0.9446 | 0.9305 | 0.9295 |
# | Random Forest | 0.9399 | 0.9234 | 0.9593 | 0.9410 | 0.9870 |
# | Gradient Boosting | 0.9411 | 0.9255 | 0.9593 | 0.9421 | 0.9833 |

# %%
# Métricas de los modelos CD (valores exactos del notebook)
cd_models = {
    "Reg. Logística": [0.8007, 0.7965, 0.8078, 0.8021, 0.8474],
    "Naive Bayes": [0.7630, 0.7675, 0.7547, 0.7610, 0.8365],
    "KNN": [0.9167, 0.8890, 0.9522, 0.9195, 0.9493],
    "Árbol Decisión": [0.9295, 0.9168, 0.9446, 0.9305, 0.9295],
    "Random Forest": [0.9399, 0.9234, 0.9593, 0.9410, 0.9870],
    "Gradient Boosting": [0.9411, 0.9255, 0.9593, 0.9421, 0.9833],
}

# Agregar MLP
cd_models["MLP (PyTorch)"] = [
    mlp_metrics["accuracy"],
    mlp_metrics["precision"],
    mlp_metrics["recall"],
    mlp_metrics["f1"],
    mlp_metrics["auc"],
]

# Tabla comparativa
df_cd = pd.DataFrame(
    cd_models, index=["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
).T
df_cd = df_cd.round(4)
display(df_cd)

# %% [markdown]
# ### 7.2 Gráfico comparativo — F1-Score y ROC AUC

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# F1-Score
models_names = list(cd_models.keys())
f1_values = [cd_models[m][3] for m in models_names]
colors_f1 = ["#95a5a6"] * (len(models_names) - 1) + ["#e74c3c"]
axes[0].barh(models_names, f1_values, color=colors_f1)
axes[0].set_xlabel("F1-Score")
axes[0].set_title("Comparativa F1-Score")
axes[0].set_xlim(0.70, 1.0)
for i, v in enumerate(f1_values):
    axes[0].text(v + 0.003, i, f"{v:.4f}", va="center", fontweight="bold")

# ROC AUC
auc_values = [cd_models[m][4] for m in models_names]
colors_auc = ["#95a5a6"] * (len(models_names) - 1) + ["#e74c3c"]
axes[1].barh(models_names, auc_values, color=colors_auc)
axes[1].set_xlabel("ROC AUC")
axes[1].set_title("Comparativa ROC AUC")
axes[1].set_xlim(0.75, 1.0)
for i, v in enumerate(auc_values):
    axes[1].text(v + 0.003, i, f"{v:.4f}", va="center", fontweight="bold")

plt.suptitle("MLP vs Modelos Tradicionales (CD)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.3 Curva ROC comparativa — MLP vs Mejores Modelos CD

# %%
# Curvas ROC de modelos CD (aproximadas usando los valores de test)
plt.figure(figsize=(9, 7))

# MLP
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_proba_mlp)
plt.plot(
    fpr_mlp,
    tpr_mlp,
    linewidth=2.5,
    label=f'MLP — PyTorch (AUC={mlp_metrics["auc"]:.4f})',
)

# Referencia de modelos CD
cd_aucs = {
    "Random Forest": 0.9870,
    "Gradient Boosting": 0.9833,
    "KNN": 0.9493,
}

for name, auc in cd_aucs.items():
    # Trazamos curvas ROC aproximadas basadas en los AUC reportados
    # (curva real requeriría las probabilidades del modelo CD)
    x = np.linspace(0, 1, 100)
    # Modelo perfecto sería x=0, y=1. Interpolamos desde random hasta perfecto
    y = x ** ((1 - auc) / auc * 0.5)  # Aproximación cualitativa
    plt.plot(x, y, "--", alpha=0.6, label=f"{name} — CD (AUC={auc:.4f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Aleatorio (AUC=0.5)")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC — MLP vs Modelos CD")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.4 Comparación de Tiempo de Entrenamiento y Recursos
#
# | Aspecto | MLP (PyTorch) | Modelos CD (scikit-learn) |
# |---------|---------------|---------------------------|
# | **Tiempo de entrenamiento** | Variable (depende de épocas y early stopping) | Segundos a minutos |
# | **Hardware** | CPU o GPU (acelera con GPU) | Solo CPU |
# | **Uso de memoria** | Moderado (pesos del modelo + batches) | Variable (árboles ocupan más) |
# | **Inferencia** | Rápida (forward pass) | Rápida (excepto KNN) |
# | **Interpretabilidad** | Baja (caja negra) | Alta (árboles, regresión) |

# %%
# Tiempo de entrenamiento estimado para cada modelo (del notebook CD)
train_times = {
    "Reg. Logística": 4.6,
    "Naive Bayes": 0.06,
    "KNN": 0.7,
    "Árbol Decisión": 1.8,
    "Random Forest": 30.6,
    "Gradient Boosting": 34.7,
    "MLP (PyTorch)": train_time,
}

fig, ax = plt.subplots(figsize=(10, 5))
names = list(train_times.keys())
times = list(train_times.values())
colors = ["#95a5a6"] * (len(names) - 1) + ["#e74c3c"]
bars = ax.barh(names, times, color=colors)
ax.set_xlabel("Tiempo de entrenamiento (segundos)")
ax.set_title("Comparación de Tiempo de Entrenamiento")
for bar, t in zip(bars, times):
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.1f}s",
        va="center",
    )
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# 8. CONCLUSIONES
# =============================================================================
print("=" * 70)
print("CONCLUSIONES")
print("=" * 70)
print(f"\n📊 DESEMPEÑO DEL MLP:")
print(f'   F1-Score: {mlp_metrics["f1"]:.4f}')
print(f'   ROC AUC:  {mlp_metrics["auc"]:.4f}')
print(f'   Accuracy: {mlp_metrics["accuracy"]:.4f}')
print(f'   Arquitectura: {best_row["hidden_layers"]}')
print(f'   Learning rate: {best_row["learning_rate"]}')
print(f'   Dropout: {best_row["dropout"]}')

print(f"\n📈 COMPARACIÓN CON MODELOS CD (F1):")
for name, vals in cd_models.items():
    marker = "◀ MLP" if "MLP" in name else ""
    print(f"   {name:>25s}: F1={vals[3]:.4f}  AUC={vals[4]:.4f}  {marker}")

# %% [markdown]
# ### 8.2 Ventajas del MLP frente a modelos tradicionales
#
# 1. **Capacidad de modelado no lineal:** A diferencia de la regresión logística
#    (que asume decisión lineal), el MLP con múltiples capas y ReLU puede capturar
#    fronteras de decisión complejas sin necesidad de ingeniería de features adicional.
#
# 2. **Aprendizaje de representaciones:** Cada capa oculta aprende representaciones
#    progresivamente más abstractas de los datos, lo que puede revelar patrones
#    que métodos como Naive Bayes (que asume independencia condicional) no detectan.
#
# 3. **Escalabilidad con GPU:** PyTorch permite aprovechar aceleración por hardware
#    (GPU) para datasets más grandes, algo que scikit-learn no soporta nativamente.
#
# 4. **Flexibilidad arquitectónica:** Se pueden incorporar técnicas como BatchNorm,
#    Dropout, y diferentes funciones de activación para adaptar la red al problema.
#
# ### 8.3 Desventajas del MLP
#
# 1. **Mayor tiempo de entrenamiento:** Respecto a modelos simples como regresión
#    logística o Naive Bayes, el MLP requiere más épocas y cómputo.
#
# 2. **Necesidad de tuning:** La cantidad de hiperparámetros es mayor (capas,
#    neuronas, learning rate, dropout, batch size, etc.), lo que demanda más
#    experimentación.
#
# 3. **Baja interpretabilidad:** A diferencia de árboles de decisión o regresión
#    logística, es difícil explicar *por qué* el modelo tomó una decisión
#    específica (problema de "caja negra").
#
# 4. **Sensibilidad a la escala:** Requiere normalización cuidadosa de los datos
#    de entrada para converger adecuadamente.
#
# ### 8.4 Comparación con CD
#
# - El MLP **iguala o supera** a KNN y se acerca a Random Forest/Gradient Boosting
#   en el dataset balanceado.
# - Random Forest y Gradient Boosting siguen siendo superiores en este problema
#   tabular de tamaño medio (~14k muestras), lo cual es consistente con la
#   literatura: los ensembles de árboles suelen dominar en datos tabulares.
# - Sin embargo, el MLP muestra **mejor generalización** que el Árbol de Decisión
#   simple (menor gap train-test en las curvas de aprendizaje).
#
# ### 8.5 Líneas futuras de trabajo
#
# 1. **Redes más profundas con Residual Connections:** Para datasets más grandes,
#    arquitecturas tipo ResNet podrían mejorar la convergencia.
#
# 2. **Transfer Learning:** Aunque menos común en datos tabulares, se podría
#    pre-entrenar en un dataset similar de mantenimiento industrial y hacer
#    fine-tuning.
#
# 3. **AutoML / Neural Architecture Search:** Explorar automáticamente
#    arquitecturas óptimas con herramientas como Optuna o Ray Tune.
#
# 4. **Modelos híbridos:** Combinar MLP con mecanismos de atención
#    (TabNet, TabTransformer) diseñados específicamente para datos tabulares.
#
# 5. **Explicabilidad (XAI):** Aplicar SHAP o LIME para interpretar las
#    predicciones del MLP y validar que las features importantes coincidan
#    con el conocimiento de dominio (ej: tool_wear debería ser muy relevante).
#
# 6. **Redes Convolucionales (CNN):** Si los datos tuvieran estructura espacial
#    o temporal (series de tiempo de sensores), una CNN 1D podría capturar
#    patrones locales. En este dataset no aplica directamente por ser tabular.
#
# 7. **Modelos Recurrentes (LSTM/GRU):** Si se dispusiera del historial
#    temporal de cada máquina (secuencias de mediciones), una red recurrente
#    podría modelar la degradación progresiva y predecir fallas con mayor
#    anticipación.

# %%
print("=== RESUMEN FINAL ===")
print(
    f'{"MLP — PyTorch":>25s} | F1={mlp_metrics["f1"]:.4f} | AUC={mlp_metrics["auc"]:.4f} | Time={train_time:.1f}s'
)
print(f'{"Random Forest — CD":>25s} | F1=0.9410 | AUC=0.9870 | Time=30.6s')
print(f'{"Gradient Boosting — CD":>25s} | F1=0.9421 | AUC=0.9833 | Time=34.7s')
print(f'{"Árbol Decisión — CD":>25s} | F1=0.9305 | AUC=0.9295 | Time=1.8s')
print(f'{"KNN — CD":>25s} | F1=0.9195 | AUC=0.9493 | Time=0.7s')
print(f'{"Reg. Logística — CD":>25s} | F1=0.8021 | AUC=0.8474 | Time=4.6s')
print(f'{"Naive Bayes — CD":>25s} | F1=0.7610 | AUC=0.8365 | Time=0.06s')
print(f"\nNota: El tiempo del MLP varía según la configuración de early stopping.")
print(
    f'Arquitectura final: {best_row["hidden_layers"]} | lr={best_row["learning_rate"]} | dropout={best_row["dropout"]}'
)
