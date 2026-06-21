import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import skimage
import optuna
import skdim

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = "/media/laurids/Elements/microclot_images/June_2026_filtered_csv"
SAVE_DIR = "final_logistic"
os.makedirs(SAVE_DIR, exist_ok=True)

N_QUANTILES = 10
MIN_AREA_THRESHOLD = 20
EXCLUDE_SPHERICAL = True
COMPACTNESS_TOLERANCE = 0.01

MODEL = LogisticRegression(
    class_weight='balanced',
    solver='lbfgs',   
    C=1.0,
    max_iter=10000,
    random_state=42
)

N_FIXED   = 64      # subsample EVERY file to this many points -> estimates are comparable
K1, K2    = 10, 40  # GP neighbor range for the slope; must satisfy K1 < K2 < N_FIXED
N_RESAMPLE = 10     # average the estimate over this many subsamples (variance control)
assert K1 < K2 < N_FIXED

def correlation_dimension(coords, seed=42):
    """Correlation dimension of one file's point cloud, size-controlled.
    Returns NaN if the file has fewer than N_FIXED points."""
    coords = np.asarray(coords, dtype=float)
    coords = coords[np.isfinite(coords).all(axis=1)]   # drop rows with NaN/inf
    n = len(coords)
    if n < N_FIXED:
        return np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_RESAMPLE):
        sub = coords[rng.choice(n, N_FIXED, replace=False)]
        try:
            vals.append(skdim.id.CorrInt(k1=K1, k2=K2).fit(sub).dimension_)
        except Exception:
            continue
    return float(np.mean(vals)) if vals else np.nan

def load_patient_data(data_dir, 
                      filenames = None,
                      min_area_threshold=None, 
                      exclude_spherical=True, 
                      compactness_tolerance=0.05,
                      drop_list=None,
                      exclude_poor_masks=True): 
    """
    Load and aggregate patient data from CSV files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files
    min_area_threshold : float, optional
        Minimum area threshold for filtering objects
    exclude_spherical : bool
        Whether to exclude spherical objects based on compactness
    compactness_tolerance : float
        Tolerance for compactness filtering (default 0.1)
    drop_list : list, optional
        List of feature indices to exclude (passed to extract_features)
    
    Returns:
    --------
    dict : patient_dict with structure:
        {patient_id: {'label': int, 'features': {feature_name: value}}}
    """
    patient_dict = {}

    if filenames is None:
        filenames = os.listdir(data_dir)

    filenames = sorted(filenames)
    pid_list = []
    pid_all = []

    for fname in tqdm(filenames):
        if not fname.endswith(".csv"):
            continue

        if "2402" in fname: # 51 year old subject
            continue
        
        df = pd.read_csv(os.path.join(data_dir, fname))
        if df.empty:
            continue
        
        # Apply filters
        if min_area_threshold is not None and 'area' in df.columns:
            df = df[df['area'] >= min_area_threshold]
        
        if exclude_spherical and 'compactness' in df.columns:
            df = df[np.abs(df['compactness'] - 1.0) > compactness_tolerance]

        if np.min(df['end_dist']) < -0.1 or np.max(df['end_dist']) > 1.1 or np.min(df['wall_dist']) < -0.1 or np.max(df['wall_dist']) > 1.1:
            continue

        pid = get_patient_id(fname)
        feats = extract_features(df, drop_list=drop_list)
        pid_all.append(pid)

        if exclude_poor_masks:

            base = fname.split("output")[0]
            primary = os.path.join(data_dir, base + "_mask.tif")
            fallback = os.path.join(data_dir, base + "mask.tif")

            try:
                mask_img = skimage.io.imread(primary)
            except Exception:
                try:
                    mask_img = skimage.io.imread(fallback)
                except Exception as e:
                    print(f"Could not read mask for {fname}: tried {primary} and {fallback} ({e})")
                    continue

            boundary_pixels = mask_img.shape[1] // 10
            mask_img = mask_img.astype(bool)
            vals = mask_img[:, :boundary_pixels].sum() + mask_img[:, mask_img.shape[1]-boundary_pixels:].sum()

        pid_list.append(pid)
        label = get_group(pid)

        if label == -1:
            continue

        if exclude_poor_masks:

            # filter failed masks. if mask coverage is less than 25% of the image, it didn't succeed
            # also no success if masks are touching the image rim or too close to it
            if mask_img.sum() / mask_img.size < 0.25 or vals > 0:
                cont_fail += 1
                print(fname)
                continue
        
        # Initialize patient entry if needed
        if pid not in patient_dict:
            patient_dict[pid] = {'label': label, 'features': {}}
        
        # Accumulate features across images for this patient
        for k, v in feats.items():
            if np.isscalar(v):
                patient_dict[pid]['features'].setdefault(k, []).append(v)
            else:
                patient_dict[pid]['features'].setdefault(k, [])
                patient_dict[pid]['features'][k].append(v)
    
    for pid, pdata in patient_dict.items():
        for k, v in pdata['features'].items():
            if isinstance(v, list):
                if np.isscalar(v[0]):
                    pdata['features'][k] = np.median(v)
                else:
                    pdata['features'][k] = np.concatenate(v)
    
    return patient_dict


def filter_compactness_values(patient_dict, compactness_tolerance):
    out = {}
    for name, data in patient_dict.items():
        feats = data['features']
        comp = np.asarray(feats['compactness'])
        mask = np.abs(comp - 1.0) > compactness_tolerance
        new_feats = {
            key: (np.asarray(vals)[mask]
                  if np.asarray(vals).shape == mask.shape
                  else vals)
            for key, vals in feats.items()
        }
        out[name] = {**data, 'features': new_feats}
    return out

def get_group(pid):
    if 'R' in pid:       # recovered → filter out
        return -1
    if 'LC' in pid:      # long covid
        return 0
    return 1             # healthy control

def get_patient_id(filename):
    filename = filename.replace("__", "_")
    base = os.path.basename(filename)
    base = re.sub(r'_output_summary\.csv$', '', base)
    # Replace version style _V<number> with just _<number>
    base = re.sub(r'_V(\d+)$', r'_\1', base)
    # Find all numeric blocks
    numbers = re.findall(r'\d+', base)
    if len(numbers) > 1:
        # Remove the last numeric block (assumed image index)
        # Find last numeric block position
        last_num = numbers[-1]
        # Remove last numeric block (and underscores before it)
        base = re.sub(r'(_+)' + re.escape(last_num) + r'$','', base)

    # Clean any trailing underscores left behind
    base = re.sub(r'_+$', '', base)
    return base

def extract_features(df, drop_list=None):

    features = {}
    # count
    if drop_list is None or 0 not in drop_list:
        features['total_clot_count'] = len(df)

    # per-object vectors
    vector_cols = [
        'area',          # 1
        'elongation',    # 2
        'axis_ratio',    # 3
        'compactness',   # 4
        'thickness_var'  # 5
    ]

    for i, c in enumerate(vector_cols, start=1):
        if drop_list is None or i not in drop_list:
            if c in df.columns:
                features[c] = df[c].dropna().to_numpy()
            else:
                features[c] = np.array([])

    pos_features = ['wall_dist', 'end_dist']  # 6 and 7
    pos = [] # pool 2D position as 1D feature vector to facilitate quantile vector aggregation
    ########## like so [wall_1…wall_N, end_1…end_N]
    for j, c in enumerate(pos_features, start=6): 
        if drop_list is None or j not in drop_list: 
            if c in df.columns: 
                pos.extend(df[c].dropna().to_numpy()) 
                if drop_list is None or not any(k in drop_list for k in [6, 7]): 
                    features['position'] = np.asarray(pos)

    return features

def drop_features(patients, drop_list=None):
    if not drop_list:
        return patients

    index_to_key = {
        0: 'total_clot_count',
        1: 'area',
        2: 'elongation',
        3: 'axis_ratio',
        4: 'compactness',
        5: 'thickness_var',
        6: 'position',  # pooled wall_dist
        7: 'position',  # pooled end_dist
    }

    keys_to_drop = {index_to_key[i] for i in drop_list if i in index_to_key}
    return {
        pid: {
            **entry,
            'features': {k: v for k, v in entry['features'].items() if k not in keys_to_drop},
        }
        for pid, entry in patients.items()
    }

def patient_quantiles(patient_dict, n_quantiles):
    rows = []

    for pid, pdata in patient_dict.items():
        row = {'patient_id': pid, 'label': pdata['label']}

        for feat, val in pdata['features'].items():
            if np.isscalar(val):
                row[feat] = val
            else:
                if len(val) == 0:
                    for q in range(n_quantiles):
                        row[f"{feat}_q{q}"] = 0.0
                else:
                    qs = np.percentile(val, np.linspace(0, 100, n_quantiles))
                    for i, v in enumerate(qs):
                        row[f"{feat}_q{i}"] = v

        rows.append(row)

    return pd.DataFrame(rows).fillna(0)

# ============================================================================
# OPTUNA FEATURE SELECTION (uncomment to run)
# ============================================================================

# base_patient_dict = load_patient_data(
#     data_dir=DATA_DIR,
#     min_area_threshold=MIN_AREA_THRESHOLD,
#     exclude_spherical=False,
#     compactness_tolerance=COMPACTNESS_TOLERANCE,
#     drop_list=[],
# )

# def evaluate(patient_dict, quant):
#     X_df = patient_quantiles(patient_dict, quant)
#     y = X_df['label'].to_numpy()
#     X_df = X_df.drop(columns=['patient_id', 'label'])

#     loo = LeaveOneOut()
#     y_true_all, y_pred_all = [], []
#     for train_idx, test_idx in loo.split(X_df, y):
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_df.iloc[train_idx])
#         X_test = scaler.transform(X_df.iloc[test_idx])
#         MODEL.fit(X_train, y[train_idx])
#         preds = MODEL.predict(X_test)
#         y_true_all.append(y[test_idx][0])
#         y_pred_all.append(preds[0])

#     return f1_score(y_true_all, y_pred_all, average='macro')

# def objective(trial):
#     compact = trial.suggest_float('compact', 0.01, 0.29, step=0.01)
#     quant   = trial.suggest_int('quant', 2, 29)
#     drop_indices = [i for i in range(8)
#                     if trial.suggest_categorical(f'drop__{i}', [False, True])]
#     if len(drop_indices) > 6:
#         raise optuna.TrialPruned()

#     pd_now = filter_compactness_values(base_patient_dict, compact)  
#     pd_now = drop_features(pd_now, drop_indices)                      
#     return evaluate(pd_now, quant)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=1000, show_progress_bar=True)

# best = study.best_params
# best_drop_list = [i for i in range(8) if best[f'drop__{i}']]
# print("Best drop list (indices):", best_drop_list)

# # optional: show the actual feature keys those map to
# index_to_key = {
#     0: 'total_clot_count', 1: 'area', 2: 'elongation', 3: 'axis_ratio',
#     4: 'compactness', 5: 'thickness_var', 6: 'position', 7: 'position',
# }
# print("Best drop list (keys):", {index_to_key[i] for i in best_drop_list})
# print("Best macro F1:", study.best_value)
# print("Best quant:", best['quant'])
# print("Best compact:", best['compact'])
#[1, 2, 5]

# ============================================================================
# LOAD + AGGREGATE
# ============================================================================

patient_dict = load_patient_data(data_dir=DATA_DIR,
                                 min_area_threshold=MIN_AREA_THRESHOLD,
                                 exclude_spherical=False,
                                 compactness_tolerance=COMPACTNESS_TOLERANCE,
                                 drop_list=[1, 2, 5]
                                 )

# ============================================================================
# BUILD DESIGN MATRIX
# ============================================================================

X_df = patient_quantiles(patient_dict, 10)
y = X_df['label'].to_numpy()
names = X_df['patient_id'].tolist()
X_df = X_df.drop(columns=['patient_id', 'label'])

#============================================================================
# CROSS VALIDATION
# ============================================================================

loo = LeaveOneOut()

y_true_all = []
y_pred_all = []
all_coef = []
all_probs = []

for train_idx, test_idx in loo.split(X_df, y):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_df.iloc[train_idx])
    X_test = scaler.transform(X_df.iloc[test_idx])

    y_train = y[train_idx]
    y_test = y[test_idx]

    MODEL.fit(X_train, y_train)

    preds = MODEL.predict(X_test)
    probs = MODEL.predict_proba(X_test)

    y_true_all.append(y_test[0])
    y_pred_all.append(preds[0])
    all_probs.append(probs[0])

# ============================================================================
# RESULTS (OVER ALL FOLDS)
# ============================================================================

print("\n" + "="*70)
print("FINAL PERFORMANCE (PATIENT LEVEL)")
print("="*70)

report_dict = classification_report(
    y_true_all,
    y_pred_all,
    target_names=['Long Covid', 'HC'],
    output_dict=True
)

print(classification_report(
    y_true_all,
    y_pred_all,
    target_names=['Long Covid', 'HC']
))

cm = confusion_matrix(y_true_all, y_pred_all, labels=[1, 0])
ConfusionMatrixDisplay(cm, display_labels=["HC", "Long Covid"]).plot(cmap="Blues", values_format="d")
plt.show()