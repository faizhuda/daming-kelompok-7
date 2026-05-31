# CLAUDE.md — Panduan untuk AI Assistant

File ini berisi konteks, konvensi, dan panduan penting untuk membantu AI assistant memahami dan bekerja secara efektif di dalam repositori ini.

---

## Ringkasan Proyek

**Nama Proyek:** Prediksi AQI Bangladesh — Daming Kelompok 7  
**Tujuan:** Membangun pipeline *Machine Learning* dan *Time-Series Forecasting* untuk memprediksi nilai **AQI (Air Quality Index)** secara per-jam menggunakan data sensor multi-polutan dari kota-kota besar Bangladesh.  
**Dataset:** `AQI Bangladesh.csv` (~94 MB, ~1,048,551 baris, 13 kolom), berisi data per jam dari 30+ kota sejak tahun 2000 hingga 2025.

**Anggota Tim:**
- Steven Lie Wibowo — G6401231021
- Tristian Yosa — G6401231122
- Faiz Naufal Huda — G6401231124
- Daffa Naufal Mumtaz — G6401231168

---

## Struktur Repositori

```
daming-kelompok-7/
├── .github/workflows/
│   └── python-app.yml       # CI/CD: flake8 + black + pytest --cov
├── configs/
│   └── config.yaml          # Parameter terpusat (kolom, lag, rolling window, model hyperparams)
├── notebooks/               # Pipeline eksplorasi & eksperimen
│   ├── .gitignore           # Mengabaikan data/, results/, models/, artifacts/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis (EDA)
│   ├── 02_cleaning.ipynb    # Data Cleaning → simpan data/df_clean.csv
│   ├── 03_feature_engineering.ipynb  # Feature Engineering → simpan data/df_feat.csv
│   ├── 04_preprocessing.ipynb        # Encoding + Scaling → simpan data/processed/ + artifacts/
│   ├── 05_modelling.ipynb   # Training XGBoost + 2 baselines → simpan models/ + results/
│   ├── 06_model_evaluation.ipynb  # Deep evaluation: residual, per-city, CV, feature importance
│   └── 07_lstm_forecasting.ipynb  # LSTM Bidirectional — forecasting AQI 30 hari ke depan
├── src/                     # Kode modular yang dapat diimpor
│   ├── cleaning.py          # Fungsi pembersihan data
│   ├── config_loader.py     # Pembaca konfigurasi YAML
│   ├── features.py          # Fungsi ekstraksi fitur
│   ├── models.py            # Fungsi pelatihan model (logging, config-driven)
│   ├── pipeline.py          # CLI end-to-end pipeline (clean → features)
│   └── utils.py             # Evaluasi, plotting, validate_columns, log_experiment
├── tests/
│   ├── conftest.py              # Fixtures pytest (dummy data + sample_config)
│   ├── test_cleaning.py         # Unit test untuk src/cleaning.py
│   ├── test_config_loader.py    # Unit test untuk src/config_loader.py
│   ├── test_features.py         # Unit test untuk src/features.py
│   ├── test_models.py           # Unit test untuk src/models.py
│   ├── test_pipeline.py         # Unit test untuk src/pipeline.py (run_clean, run_features, main)
│   └── test_utils.py            # Unit test untuk src/utils.py
├── AQI Bangladesh.csv       # Dataset mentah (jangan diubah)
├── Dockerfile               # Docker image Python 3.11 + Jupyter Lab
├── docker-compose.yml       # Jalankan Jupyter dengan `docker compose up`
├── .dockerignore            # File yang dikecualikan saat build Docker image
├── .pre-commit-config.yaml  # black + flake8 otomatis sebelum commit
├── requirements.txt         # Dependensi Python (versioned dengan upper bounds >=X,<Y)
└── README.md                # Dokumentasi utama
```

---

## Perintah Umum (Common Commands)

Semua perintah dijalankan dari **root direktori proyek** dengan virtual environment `.venv` aktif.

### Setup Environment
```bash
# Buat virtual environment dengan Python 3.11 (direkomendasikan, sesuai CI)
py -3.11 -m venv .venv

# Aktivasi (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install semua dependensi
pip install -r requirements.txt
```

### Menjalankan Unit Test
```bash
# Jalankan semua test dengan coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Jalankan test spesifik
python -m pytest tests/test_cleaning.py -v
python -m pytest tests/test_config_loader.py -v
python -m pytest tests/test_features.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_utils.py -v
```

### Kualitas Kode
```bash
# Format kode dengan black
black src/ tests/

# Cek format tanpa mengubah (mode CI)
black --check src/ tests/

# Lint dengan flake8 (cek error kritis saja)
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Lint lengkap (warning mode)
flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### Pre-commit Hooks (Jalankan Sekali Setelah Clone)
```bash
# Aktifkan hooks — black + flake8 akan berjalan otomatis sebelum setiap commit
pre-commit install

# Jalankan manual pada semua file (opsional)
pre-commit run --all-files
```

### Menjalankan Pipeline via CLI (Tanpa Jupyter)
```bash
# Jalankan seluruh pipeline: cleaning → feature engineering
python src/pipeline.py --stage all

# Atau per tahap
python src/pipeline.py --stage clean
python src/pipeline.py --stage features

# Gunakan input/output kustom
python src/pipeline.py --stage all --input "AQI Bangladesh.csv" --output-dir notebooks/data
```

### Menjalankan Jupyter via Docker
```bash
# Build image dan jalankan container (buka http://localhost:8888)
docker compose up

# Jalankan di background
docker compose up -d
```

### Menjalankan Notebooks (Urutan Pipeline)

> **PENTING:** Notebook-notebook ini merupakan sebuah pipa data (*data pipeline*) yang harus dijalankan **secara berurutan**. Setiap notebook menyimpan hasilnya ke disk agar notebook berikutnya dapat memuatnya secara mandiri (tanpa memerlukan sesi Kernel yang sama).

```
01_eda.ipynb             (opsional, hanya untuk eksplorasi)
→ 02_cleaning.ipynb      (wajib pertama kali, menghasilkan notebooks/data/df_clean.csv)
→ 03_feature_engineering.ipynb  (menghasilkan notebooks/data/df_feat.csv)
→ 04_preprocessing.ipynb        (menghasilkan notebooks/data/processed/ & notebooks/artifacts/)
→ 05_modelling.ipynb            (menghasilkan notebooks/models/ & notebooks/results/)
→ 06_model_evaluation.ipynb    (evaluasi mendalam: residual, per-city, walk-forward CV)

07_lstm_forecasting.ipynb      (standalone — membaca AQI Bangladesh.csv langsung, tidak bergantung pada nb02–06)
```

Jalankan Jupyter dari folder `notebooks/`:
```bash
cd notebooks
jupyter lab
```

---

## Arsitektur & Konvensi Kode

### Konfigurasi Terpusat (`configs/config.yaml`)

Semua *hardcoded value* — seperti nama kolom, batas winsorizing, daftar lag, dan ukuran rolling window — harus **selalu** diambil dari file konfigurasi ini, bukan di-*hardcode* langsung di dalam kode.

```python
from src.config_loader import load_config

config = load_config()
pollutant_cols  = config["data"]["pollutant_cols"]
lags            = config["features"]["lags"]           # [1, 3, 6, 24]
rolling_windows = config["features"]["rolling_windows"] # [3, 6, 24]

# Hyperparameter model dari config, bukan hardcoded:
xgb_cfg = config["models"]["xgboost"]  # n_estimators, max_depth, learning_rate

# Konfigurasi visualisasi:
plot_n = config["visualization"]["plot_sample_size"]  # max titik di time-series plot
```

### Modul `src/` — Konvensi Penulisan

Semua fungsi di `src/` mengikuti standar:
- **Type hints** wajib pada semua parameter dan return value.
- **Docstring** format Google Style wajib ada, mencakup `Args:` dan `Returns:`.
- Setiap fungsi menerima parameter `config: Optional[Dict[str, Any]] = None`, dan jika `None` maka memuat config dari YAML secara otomatis.
- Fungsi **selalu mengembalikan** DataFrame baru (tidak memodifikasi *in-place* tanpa sengaja).

**Contoh pola yang benar:**
```python
def create_lag_features(
    df: pd.DataFrame,
    lag_cols: List[str],
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Creates lagged features per city to prevent lookahead bias.

    Args:
        df (pd.DataFrame): Input dataframe.
        lag_cols (List[str]): Columns to create lag features for.
        config (Optional[Dict[str, Any]]): Config dict. Loads from YAML if None.

    Returns:
        pd.DataFrame: Dataframe with added lag feature columns.
    """
    if config is None:
        config = load_config()
    ...
```

### Kolom-Kolom Penting Dataset

| Kolom | Deskripsi |
|---|---|
| `city_id` | ID numerik unik per kota |
| `city_name` | Nama kota (string) |
| `datetime` | Timestamp per jam (UTC) |
| `pm10`, `pm2_5` | Particulate Matter (µg/m³) |
| `carbon_monoxide` | CO (µg/m³) |
| `nitrogen_dioxide` | NO₂ (µg/m³) |
| `sulphur_dioxide` | SO₂ (µg/m³) |
| `ozone` | O₃ (µg/m³) |
| `carbon_dioxide` | CO₂ — **di-drop** saat cleaning (>74% missing) |
| `aqi` | **TARGET** — Air Quality Index (numerik kontinu) |

### Konvensi Penamaan Fitur Turunan

| Pola Nama | Contoh | Keterangan |
|---|---|---|
| `{col}_lag{n}` | `aqi_lag1`, `pm10_lag24` | Lag n jam ke belakang |
| `{col}_roll{w}m` | `pm2_5_roll3m` | Rolling mean window w jam |
| `{col}_roll{w}std` | `aqi_roll6std` | Rolling std window w jam |
| `hour_sin`, `hour_cos` | — | Cyclical encoding jam (0–23) |
| `month_sin`, `month_cos` | — | Cyclical encoding bulan (1–12) |
| `dow_sin`, `dow_cos` | — | Cyclical encoding hari dalam seminggu |
| `pm_ratio_lag1` | — | pm2_5[t-1] / (pm10[t-1] + ε) — fraksi partikel halus |
| `pm_total_lag1` | — | pm10[t-1] + pm2_5[t-1] — beban partikel total |
| `oxidant_load_lag1` | — | NO₂[t-1] + O₃[t-1] — beban oksidan |
| `combustion_idx_lag1` | — | CO[t-1] / (NO₂[t-1] + ε) — indikator pembakaran tidak sempurna |

---

## Model Machine Learning yang Digunakan

### Di `notebooks/05_modelling.ipynb`

Model utama adalah **XGBoost Regressor**, dipilih berdasarkan studi Grinsztajn et al. (NeurIPS 2022) yang menunjukkan gradient boosting secara konsisten unggul atas LSTM pada tabular data dengan fitur yang sudah di-rekayasa secara eksplisit.

| Model | Jenis | Catatan |
|---|---|---|
| **Baseline Naive** | Statistik | y[t] = y[t-1]; kompetitif di MAE karena autokorelasi AQI tinggi |
| **Baseline Rolling Mean (24h)** | Statistik | Rolling mean 24 jam terakhir dari data train |
| **XGBoost** | Gradient Boosting | Early stopping (patience=20 rounds); 500 trees max, eval on val set |

> **Catatan:** LSTM, ARIMA, dan Prophet tidak digunakan dalam proyek ini. Model yang diimplementasikan hanyalah XGBoost sebagai model utama dan dua baseline statistik.

### Di `notebooks/06_model_evaluation.ipynb` (Deep Evaluation)

| Analisis | Deskripsi |
|---|---|
| Tabel metrik | MAE/RMSE/R²/MAPE semua model vs baseline |
| Feature importance | Top 25 fitur XGBoost (gain-based) |
| Residual analysis | 4-panel: histogram, vs waktu, vs prediksi, Q-Q plot |
| Per-city breakdown | MAE/RMSE per kota, sorted; scatter mean AQI vs MAE |
| Extreme AQI (>150) | Bias, MAE khusus kejadian ekstrem vs normal |
| Walk-Forward CV | 3-fold `TimeSeriesSplit` pada training set |

### Di `src/models.py` (Fungsi Modular)
| Fungsi | Deskripsi |
|---|---|
| `train_xgboost()` | XGBoost Regressor dengan early stopping (config-driven) |
| `create_sequences()` | Konversi array 2D ke input sekuensial 3D untuk model berbasis sequence |

---

## Pipeline Data Notebook — Berkas Intermediate

File-file berikut **dibuat secara otomatis** saat notebook dieksekusi berurutan. File ini **tidak di-commit ke Git** (dicantumkan di `notebooks/.gitignore`).

```
notebooks/
├── data/
│   ├── df_clean.csv                   # Keluaran 02_cleaning.ipynb
│   ├── df_feat.csv                    # Keluaran 03_feature_engineering.ipynb
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── df_train_datetime.csv
│       └── df_test_datetime.csv
├── artifacts/
│   ├── robust_scaler.pkl              # Keluaran 04_preprocessing.ipynb
│   ├── label_encoder_city.pkl
│   ├── feature_list.json
│   ├── lstm_scaler_X.pkl              # Keluaran 07_lstm_forecasting.ipynb (RobustScaler fitur)
│   └── lstm_scaler_y.pkl              # Keluaran 07_lstm_forecasting.ipynb (MinMaxScaler target)
├── models/
│   ├── xgboost_model.pkl              # Keluaran 05_modelling.ipynb
│   ├── best_lstm_aqi.h5               # Keluaran 07_lstm_forecasting.ipynb (best checkpoint)
│   └── lstm_aqi_model.h5              # Keluaran 07_lstm_forecasting.ipynb (final model)
└── results/
    ├── metrics_comparison.csv
    ├── deep_metrics_comparison.csv    # Keluaran 06_model_evaluation.ipynb
    ├── feature_importance.csv
    ├── per_city_metrics.csv
    ├── xgb_cv_results.csv
    ├── forecast_aqi_30hari.csv        # Keluaran 07_lstm_forecasting.ipynb (harian)
    ├── forecast_aqi_hourly_30hari.csv # Keluaran 07_lstm_forecasting.ipynb (per jam)
    ├── experiment_log.csv             # Log semua training run (log_experiment)
    └── plots/
        ├── xgboost_prediction.png
        ├── metrics_comparison_bar.png
        ├── feature_importance_xgb.png
        ├── residual_analysis.png
        ├── per_city_performance.png
        ├── extreme_aqi_error_analysis.png
        ├── aqi_autocorrelation.png
        └── shap_analysis.png
```

---

## Unit Testing

Test fixtures tersedia di `tests/conftest.py`:
- **`sample_raw_data`** — DataFrame dummy 48 jam data sensor 1 kota (Dhaka) dengan duplikasi, nilai negatif, dan missing value.
- **`sample_config`** — Config dictionary lengkap termasuk section `models` dengan nilai kecil untuk mempercepat test (misalnya `n_estimators: 50`).

```python
# Pola penggunaan fixture dalam test
def test_clean_data_drops_negative(sample_raw_data, sample_config):
    cleaned = clean_data(sample_raw_data, config=sample_config)
    assert (cleaned["pm10"] >= 0).all()

# Test untuk utils
def test_evaluate_model_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = evaluate_model(y_true, y_pred, "PerfectModel")
    assert result["MAE"] == pytest.approx(0.0)
```

File test yang tersedia:
| File | Modul yang Ditest |
|---|---|
| `test_cleaning.py` | `src/cleaning.py` — clean_data, winsorize_city, impute_missing_linear |
| `test_config_loader.py` | `src/config_loader.py` — load_config, validasi struktur config |
| `test_features.py` | `src/features.py` — lag, rolling, cyclical, interactions |
| `test_models.py` | `src/models.py` — create_sequences, train_xgboost |
| `test_pipeline.py` | `src/pipeline.py` — run_clean, run_features, main (CLI) |
| `test_utils.py` | `src/utils.py` — evaluate_model, validate_columns, log_experiment, plot_predictions |

---

## CI/CD (GitHub Actions)

Pipeline CI berjalan otomatis di setiap **push** dan **pull request** ke branch `main`/`master`.

**Tahapan:**
1. Setup Python 3.11
2. `pip install -r requirements.txt`
3. `flake8` — cek error kritis (syntax error, undefined name)
4. `black --check` — verifikasi format kode
5. `pytest tests/ -v --cov=src --cov-report=term-missing` — jalankan semua unit test + coverage report

> **Catatan:** Jika build CI gagal, jalankan `black src/ tests/` dan `pytest` secara lokal sebelum melakukan push ulang.

---

## Experiment Tracking

Setiap training run dapat dicatat ke CSV untuk perbandingan antar eksperimen:

```python
from src.utils import evaluate_model, log_experiment

# Setelah training selesai
metrics = evaluate_model(y_true, y_pred, "XGBoost")
log_experiment(
    model_name="XGBoost",
    params=best_config,  # dict hyperparameter yang digunakan
    metrics=metrics,
    log_path="notebooks/results/experiment_log.csv"
)
```

File `experiment_log.csv` akan dibuat otomatis jika belum ada, dan setiap run baru akan ditambahkan sebagai baris baru. Kolom: `timestamp`, `model`, `params`, `mae`, `rmse`, `r2`, `mape`.

---

## Hal-Hal Penting yang Perlu Diperhatikan

1. **Python 3.11 direkomendasikan** untuk environment lokal — sesuai dengan versi yang digunakan di CI/CD.
2. **Jangan edit `src/` langsung di Google Colab** — lakukan perubahan di lokal dan push ke GitHub.
3. **Urutan notebook wajib dijaga** — setiap notebook bergantung pada output berkas CSV/PKL dari notebook sebelumnya *(kecuali `07_lstm_forecasting.ipynb` yang standalone dan membaca langsung dari `AQI Bangladesh.csv`).*
4. **Semua parameter konfigurasi harus dari `configs/config.yaml`** — jangan hardcode nilai apapun di `src/`, termasuk hyperparameter model.
5. **Train-Test Split selalu Time-Aware** — split berdasarkan waktu (80% awal = train, 20% akhir = test), bukan random, untuk menghindari data leakage.
6. **`RobustScaler` hanya di-fit pada data train**, lalu ditransformasi ke data test — jangan fit ulang pada test set.
7. **Lag dan rolling features selalu dibuat per kota** (`groupby city_id`) untuk menghindari kebocoran lintas kota.
8. **`validate_columns()` dipanggil otomatis** di `clean_data`, `create_lag_features`, dan `create_rolling_features` — fungsi akan melempar `ValueError` informatif jika kolom yang dibutuhkan tidak ada di DataFrame input.
9. **Gunakan `log_experiment()` setelah setiap training run** agar hasil dapat dibandingkan lintas eksperimen tanpa kehilangan data setelah kernel ditutup.
10. **Jalankan `pre-commit install` sekali setelah clone** — ini mengaktifkan hook yang menjalankan black dan flake8 otomatis sebelum setiap commit, sehingga CI tidak pernah gagal karena format.
