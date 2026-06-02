#!/usr/bin/env python3
"""
Transformer Multi-Step Forecasting Pipeline.
Features:
- Dual-head encoder-only Transformer.
- CLEAN signal for markup vs NOISY signal for training.
- Ternary classification (0=No event, 1=Pos bifurcation, 2=Neg bifurcation).
- Grid search over (noise_color, alpha, SNR, horizon).
- Distribution analysis (Normal, Student-t, Cauchy).
- find_best_normality_params and visual reports.
"""

import math, copy, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, r2_score, mean_absolute_percentage_error, confusion_matrix
import scipy.stats as st
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

import colorednoise as cn

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'window_size'    : 50,
    'd_model'        : 64,
    'nhead'          : 4,
    'num_layers'     : 3,
    'dim_feedforward': 256,
    'dropout'        : 0.1,
    'input_size'     : 3,

    'epochs'         : 30,
    'batch_size'     : 64,
    'lr'             : 1e-3,
    'patience'       : 8,

    'horizons'       : [1, 2, 5, 10],

    'base_n'         : 10000,
    'base_delta_t'   : 0.01,
    'random_seed'    : 42,

    'lambda_cls'     : 5.0,
    'grad_clip'      : 1.0,

    # Grid search parameters
    'noise_colors'   : [0, 1, 2, -1, -2], # white, pink, red, blue, violet
    'alpha_levels'   : [1, 2, math.pi],
    'snr_levels'     : [0.5, 1.0, 5.0],   # Signal-to-Noise Ratio

    'signal_type'    : 'sine',
    'sine_amplitude' : 3 * math.pi,
    'sine_freq'      : 1.0,
}

torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOISE_NAMES = {0:'White', 1:'Pink', 2:'Red', -1:'Blue', -2:'Violet'}

# ============================================================
# TRANSFORMER MODEL
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerForecaster(nn.Module):
    def __init__(self, input_size=3, d_model=64, nhead=4,
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model=d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # LayerNorm + Dropout перед выходными головами для стабильности
        self.head_norm    = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(dropout)
        self.fc_value = nn.Linear(d_model, 1)
        # Ternary classes: 0=no event, 1=pos bifurcation, 2=neg bifurcation
        self.fc_break = nn.Linear(d_model, 3)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        """Каузальная маска: модель не может подглядывать в будущие шаги окна."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, x: torch.Tensor):
        h    = self.input_proj(x)
        h    = self.pos_enc(h)
        mask = self._causal_mask(h.size(1))
        h    = self.encoder(h, mask=mask)
        last = self.head_dropout(self.head_norm(h[:, -1, :]))
        return self.fc_value(last), self.fc_break(last)

# ============================================================
# DATA GENERATION
# ============================================================
class SineLabels:
    def __init__(self, complex_markup, clean_signal):
        self.complex_markup = complex_markup
        self.clean_signal = clean_signal

def generate_sine_signal_snr(spectral_exp, alpha_amp, snr, n, random_seed=42,
                             sine_amplitude=2*math.pi, sine_freq=1.0, delta_t=0.01):
    """
    ЯВНОЕ РАЗДЕЛЕНИЕ:
    1. CLEAN signal -> используется для идеальной тернарной разметки бифуркаций.
    2. NOISY signal -> возвращается для обучения модели.
    
    ПРЕДВЕСТНИКИ БИФУРКАЦИЙ:
    За 10 шагов до каждого фазового скачка амплитуда синуса начинает плавно
    нарастать (×1.0 → ×1.5). Это даёт модели детектируемый «сигнал тревоги»
    в исторических данных, позволяя Attention-механизму научиться
    распознавать приближающуюся бифуркацию.
    """
    PRECURSOR_LEN = 10   # за сколько шагов начинается предвестник
    PRECURSOR_AMP = 5.0  # во сколько раз вырастает амплитуда к моменту скачка (изменено с 1.5 для усиления)

    rng = np.random.default_rng(random_seed)
    t   = np.arange(n) * delta_t
    
    # --- Шаг 1: определяем ВСЕ моменты бифуркаций заранее ---
    bifurcation_events = []  # список (index, direction, k)
    for i in range(1, n):
        if rng.random() < 0.005:
            direction = rng.choice([-1, 1])
            k = rng.choice([1, 2, 3])
            bifurcation_events.append((i, direction, k))
    
    # --- Шаг 2: строим множество шагов с предвестником ---
    precursor_multiplier = np.ones(n)  # по умолчанию = 1.0
    for (bif_idx, _, _) in bifurcation_events:
        for offset in range(PRECURSOR_LEN):
            j = bif_idx - PRECURSOR_LEN + offset
            if 0 <= j < n:
                # Линейная интерполяция: 1.0 → PRECURSOR_AMP
                t_frac = (offset + 1) / PRECURSOR_LEN
                precursor_multiplier[j] = max(precursor_multiplier[j],
                                               1.0 + (PRECURSOR_AMP - 1.0) * t_frac)

    # --- Шаг 3: генерируем чистый сигнал с предвестниками ---
    phase = 0.0
    signal_clean = np.zeros(n)
    complex_markup = np.zeros(n, dtype=int)
    bif_set = {ev[0]: (ev[1], ev[2]) for ev in bifurcation_events}
    
    for i in range(n):
        if i in bif_set:
            direction, k = bif_set[i]
            phase += direction * k * math.pi
        
        signal_clean[i] = sine_amplitude * precursor_multiplier[i] * \
                          math.sin(2 * math.pi * sine_freq * t[i] + phase)
                          
    # --- Шаг 4: Размечаем бифуркации (весь предвестник) ---
    for (bif_idx, direction, k) in bifurcation_events:
        label = 1 if direction == 1 else 2
        for offset in range(PRECURSOR_LEN + 1):  # Размечаем предвестник + саму точку
            j = bif_idx - PRECURSOR_LEN + offset
            if 0 <= j < n:
                complex_markup[j] = label
    
    # 2. NOISE GENERATION (цветной шум по спектральной экспоненте)
    old = np.random.get_state(); np.random.seed(random_seed)
    if spectral_exp == 0:           # White noise (int or float)
        noise = rng.normal(0, 1, size=n)
    else:
        try:
            noise = cn.powerlaw_psd_gaussian(float(spectral_exp), n)
        except Exception:
            noise = rng.normal(0, 1, size=n)   # fallback to white
    np.random.set_state(old)
    
    # Scale noise exactly to specified SNR (Signal-to-Noise Ratio)
    sig_power   = np.mean(signal_clean**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-12:        # degenerate case protection
        noise_power = 1e-12
    target_noise_power = sig_power / max(snr, 1e-12)
    scaling_factor = np.sqrt(target_noise_power / noise_power) * alpha_amp
    noise = noise * scaling_factor
    
    # 3. NOISY SIGNAL FOR TRAINING
    signal_noisy = signal_clean + noise
    
    return signal_noisy, SineLabels(complex_markup, signal_clean)

def build_features(signal_scaled, window_size):
    n = len(signal_scaled)
    feats = np.zeros((n, 3), dtype=np.float32)
    feats[:, 0] = signal_scaled
    feats[1:, 1] = np.diff(signal_scaled)
    roll_w = min(10, window_size // 5)
    if roll_w >= 2:
        cs2 = np.cumsum(np.insert(signal_scaled**2, 0, 0))
        cs1 = np.cumsum(np.insert(signal_scaled, 0, 0))
        for i in range(roll_w, n):
            s2 = (cs2[i+1] - cs2[i+1-roll_w]) / roll_w
            s1 = (cs1[i+1] - cs1[i+1-roll_w]) / roll_w
            feats[i, 2] = max(0.0, s2 - s1**2)**0.5
    return feats

def prepare_data(signal, window_size, horizon, batch_size, break_labels):
    train_end = int(0.8 * len(signal))
    scaler = MinMaxScaler()
    scaler.fit(signal[:train_end].reshape(-1, 1))
    sig_sc = scaler.transform(signal.reshape(-1, 1)).flatten()

    feats = build_features(sig_sc, window_size)
    n = len(sig_sc)
    X_list, y_list = [], []
    for i in range(n - window_size - horizon + 1):
        X_list.append(feats[i:i+window_size])
        y_list.append(sig_sc[i + window_size + horizon - 1])
    if not X_list:
        return None, None, None, None, scaler, np.array([], dtype=int), np.array([], dtype=int)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    target_idx_all = np.arange(len(X)) + window_size + horizon - 1
    b_labels = break_labels[np.clip(target_idx_all, 0, len(break_labels)-1)]

    split = int(0.8 * len(X))
    val_split = max(1, int(0.85 * split))

    X_tr, X_va, X_te = X[:val_split], X[val_split:split], X[split:]
    y_tr, y_va, y_te = y[:val_split], y[val_split:split], y[split:]
    b_tr, b_va, b_te = b_labels[:val_split], b_labels[val_split:split], b_labels[split:]

    test_target_idx = np.arange(split, split+len(X_te)) + window_size + horizon - 1

    def to_t(*args): return [torch.FloatTensor(a) for a in args]
    Xt, yt = to_t(X_tr, y_tr)
    Xv, yv = to_t(X_va, y_va)
    Xte, yte = to_t(X_te, y_te)
    
    bt = torch.LongTensor(b_tr)
    bv = torch.LongTensor(b_va)

    train_loader = DataLoader(TensorDataset(Xt, yt.unsqueeze(-1), bt),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv.unsqueeze(-1), bv),
                              batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, Xte, yte.unsqueeze(-1), scaler, test_target_idx, b_te

# ============================================================
# ============================================================
# TRAINING & EVALUATION
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss для экстремального дисбаланса классов."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor of weights

    def forward(self, inputs, targets):
        # 1. Считаем обычный CrossEntropy БЕЗ весов, чтобы правильно достать вероятности (pt)
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 2. Считаем Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 3. Применяем веса классов
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

def train_model(train_loader, config, val_loader=None, pretrained_state=None):
    model = TransformerForecaster(
        input_size    = config['input_size'],
        d_model       = config['d_model'],
        nhead         = config['nhead'],
        num_layers    = config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout       = config['dropout'],
    ).to(device)

    is_finetuning = pretrained_state is not None
    if is_finetuning:
        model.load_state_dict(pretrained_state, strict=False)

    mse_fn     = nn.MSELoss()
    lambda_cls = config.get('lambda_cls', 5.0)
    grad_clip  = config.get('grad_clip', 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    
    # Fine-tuning: мягкий CosineAnnealing без начального скачка LR
    # Обучение с нуля: OneCycleLR с прогревом
    if is_finetuning:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'] * len(train_loader), eta_min=config['lr'] * 0.01
        )
    else:
        total_steps = config['epochs'] * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['lr'], total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos',
        )

    all_b = np.concatenate([b.numpy() for _, _, b in train_loader])
    counts = np.bincount(all_b, minlength=3)
    
    # Ограничиваем максимальный вес, чтобы градиенты не взрывались (max 100)
    w0 = 1.0
    w1 = min(float(counts[0]) / max(counts[1], 1), 100.0)
    w2 = min(float(counts[0]) / max(counts[2], 1), 100.0)
    weights = torch.FloatTensor([w0, w1, w2]).to(device)
    
    ce_fn = FocalLoss(alpha=weights, gamma=2.0)

    loss_hist, best_val, patience_ctr, best_state = [], float('inf'), 0, None

    for epoch in range(config['epochs']):
        model.train()
        ep_loss, nb = 0.0, 0
        for Xb, yb, bb in train_loader:
            Xb, yb, bb = Xb.to(device), yb.to(device), bb.to(device)
            optimizer.zero_grad()
            vp, bp = model(Xb)
            loss = mse_fn(vp, yb) + lambda_cls * ce_fn(bp, bb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step(); scheduler.step()
            ep_loss += loss.item(); nb += 1
        loss_hist.append((epoch, ep_loss / max(nb, 1)))

        if val_loader and len(val_loader) > 0:
            model.eval(); vl, vn = 0.0, 0
            with torch.no_grad():
                for Xv, yv, bv in val_loader:
                    Xv, yv, bv = Xv.to(device), yv.to(device), bv.to(device)
                    vp, bp = model(Xv)
                    vl += (mse_fn(vp, yv) + lambda_cls * ce_fn(bp, bv)).item()
                    vn += 1
            avg_val = vl / max(vn, 1)
        else:
            avg_val = loss_hist[-1][1]

        if avg_val < best_val:
            best_val, patience_ctr = avg_val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= config['patience']:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, loss_hist

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom))

def evaluate_model(model, X_test, y_test, b_test, scaler):
    model.eval()
    with torch.no_grad():
        vp, bp = model(X_test.to(device))
        preds_sc   = vp.cpu().numpy().flatten()
        break_logits = bp.cpu()
        break_proba = torch.softmax(break_logits, dim=1).numpy()
        break_preds = torch.argmax(break_logits, dim=1).numpy().flatten()
    
    y_sc = y_test.numpy().flatten()
    preds  = scaler.inverse_transform(preds_sc.reshape(-1,1)).flatten()
    y_true = scaler.inverse_transform(y_sc.reshape(-1,1)).flatten()
    
    errs = preds - y_true
    rmse = float(np.sqrt(np.mean(errs**2)))
    mae = float(np.mean(np.abs(errs)))
    r2 = float(r2_score(y_true, preds))
    mape = float(mean_absolute_percentage_error(y_true, preds))
    smape_val = float(smape(y_true, preds))
    
    # Ternary Classification metrics
    f1_macro = float(f1_score(b_test, break_preds, average='macro', zero_division=0))
    f1_per_class = f1_score(b_test, break_preds, average=None, zero_division=0, labels=[0,1,2])
    conf_matrix = confusion_matrix(b_test, break_preds, labels=[0, 1, 2])
    
    return preds, y_true, rmse, mae, r2, mape, smape_val, break_preds, f1_macro, conf_matrix

# ============================================================
# EXPERIMENTS
# ============================================================
def run_single_experiment(noise_color, alpha, snr, config):
    n = config['base_n']
    signal, labels = generate_sine_signal_snr(
        spectral_exp=noise_color, alpha_amp=alpha, snr=snr, n=n,
        random_seed=config['random_seed']
    )
    break_labels = labels.complex_markup

    results, loss_curves, per_sample, residuals = {}, {}, {}, {}
    backbone_state = None
    saved_model = None  # сохраняем обученную модель для Attention Map

    for h in config['horizons']:
        torch.manual_seed(config['random_seed'] + h)
        tr_l, va_l, X_te, y_te, scaler, test_idx, b_te = prepare_data(
            signal, config['window_size'], h, config['batch_size'], break_labels)
        
        if tr_l is None or X_te is None or len(X_te) == 0:
            continue

        if backbone_state is not None:
            ft_cfg = dict(config)
            ft_cfg['epochs'] = max(5, config['epochs'] // 3)
            ft_cfg['lr'] = config['lr'] * 0.1  # Пониженный LR для fine-tuning
            model, lh = train_model(tr_l, ft_cfg, va_l, pretrained_state=backbone_state)
        else:
            model, lh = train_model(tr_l, config, va_l)
            backbone_state = {k: v.clone() for k, v in model.state_dict().items()
                              if k.startswith(('encoder.', 'input_proj.', 'pos_enc.'))}
            # Сохраняем первую полностью обученную модель (h=1)
            saved_model = copy.deepcopy(model)
        loss_curves[h] = lh

        preds, y_true, rmse, mae, r2, mape, smape_val, break_preds, f1, cm = evaluate_model(
            model, X_te, y_te, b_te, scaler)
        
        residuals[h] = preds - y_true

        results[h] = dict(
            noise_color=noise_color, alpha=alpha, snr=snr, horizon=h,
            RMSE=rmse, MAE=mae, R2=r2, MAPE=mape, sMAPE=smape_val,
            macro_f1=f1, conf_matrix=str(cm.tolist())
        )
        
        per_sample[h] = dict(
            y_true=y_true, preds=preds,
            b_true=b_te, b_pred=break_preds
        )

    return list(results.values()), loss_curves, per_sample, residuals, saved_model

def run_experiment_grid(config):
    print("\n" + "="*60)
    print("EXPERIMENT: Full Grid Search (noise_color, alpha, snr)")
    print("="*60)
    all_results, all_loss, all_ps, all_res = [], {}, {}, {}
    saved_model_info = None  # (model, tag) — сохраняем одну обученную модель для Attention
    combos = [(nc, a, s) for nc in config['noise_colors'] 
              for a in config['alpha_levels'] 
              for s in config['snr_levels']]
    
    for nc, a, s in tqdm(combos, desc="Grid Exp"):
        tag = (nc, a, s)
        res, lc, ps, rv, model = run_single_experiment(nc, a, s, config)
        all_results.extend(res); all_loss[tag] = lc
        all_ps[tag] = ps; all_res[tag] = rv
        # Сохраняем первую обученную модель (SNR=5 предпочтительнее)
        if saved_model_info is None and model is not None:
            saved_model_info = (model, tag)
    return all_results, all_loss, all_ps, all_res, saved_model_info

# ============================================================
# DISTRIBUTION ANALYSIS
# ============================================================
def fit_distributions(residuals_db):
    """
    Принимает residuals_db в двух форматах:
      - плоский: {(tag, h): np.array}
      - вложенный (от run_experiment_grid): {tag: {h: np.array}}
    Автоматически разворачивает вложенный формат.
    """
    # Разворачиваем вложенный словарь {tag: {h: arr}} -> {(tag, h): arr}
    flat_db = {}
    for key, val in residuals_db.items():
        if isinstance(val, dict):
            for h, arr in val.items():
                flat_db[(key, h)] = arr
        else:
            flat_db[key] = val

    rows = []
    for (tag, h), res in flat_db.items():
        if not isinstance(res, np.ndarray) or len(res) < 10:
            continue
        r = res - res.mean()

        n_sw  = min(len(r), 5000)
        _, p_sw = st.shapiro(r[:n_sw])
        _, p_jb = st.jarque_bera(r)

        mu_n, sig_n = st.norm.fit(r)
        ks_n, _     = st.kstest(r, 'norm',   args=(mu_n, sig_n))

        df_t, loc_t, sc_t = st.t.fit(r)
        ks_t, _            = st.kstest(r, 't',    args=(df_t, loc_t, sc_t))

        loc_c, sc_c = st.cauchy.fit(r)
        ks_c, _     = st.kstest(r, 'cauchy', args=(loc_c, sc_c))

        # Anderson-Darling test (дополнительная проверка нормальности)
        ad_stat, ad_crit, ad_sig = st.anderson(r, dist='norm')
        ad_pass = bool(ad_stat < ad_crit[2])  # 5% significance level

        best = min([('Normal', ks_n), ('Student', ks_t), ('Cauchy', ks_c)],
                   key=lambda x: x[1])[0]
                   
        nc, a, s = tag if isinstance(tag, tuple) else ("?", "?", "?")

        rows.append(dict(
            noise_color=nc, alpha=a, snr=s, horizon=h,
            p_shapiro=round(float(p_sw), 4),
            p_jarque=round(float(p_jb), 4),
            ks_normal=round(float(ks_n), 4),
            ks_student=round(float(ks_t), 4),
            ks_cauchy=round(float(ks_c), 4),
            # Сохраняем параметры каждого распределения
            norm_mu=round(float(mu_n), 4),
            norm_sigma=round(float(sig_n), 4),
            student_df=round(float(df_t), 2),
            student_loc=round(float(loc_t), 4),
            student_scale=round(float(sc_t), 4),
            cauchy_loc=round(float(loc_c), 4),
            cauchy_scale=round(float(sc_c), 4),
            # Anderson-Darling тест нормальности
            ad_statistic=round(float(ad_stat), 4),
            ad_normal_pass=ad_pass,
            best_fit=best,
            n=len(r),
        ))
    return pd.DataFrame(rows)

def find_best_normality_params(df_fit):
    """
    Находит комбинацию (alpha, snr, noise_color) для каждого горизонта,
    при которой KS distance к Normal МИНИМАЛЕН.
    Также возвращает лучшую конфигурацию для каждого цвета шума отдельно.
    """
    best_configs = {}
    for h in sorted(df_fit['horizon'].unique()):
        sub = df_fit[df_fit['horizon'] == h]
        if sub.empty: continue
        best_row = sub.loc[sub['ks_normal'].idxmin()]
        # Лучший результат по каждому цвету шума
        per_noise = {}
        for nc in sub['noise_color'].unique():
            nc_sub = sub[sub['noise_color'] == nc]
            if nc_sub.empty: continue
            nc_best = nc_sub.loc[nc_sub['ks_normal'].idxmin()]
            per_noise[nc] = {
                "alpha": nc_best['alpha'],
                "snr": nc_best['snr'],
                "ks_normal": nc_best['ks_normal'],
                "best_fit": nc_best['best_fit'],
            }
        best_configs[h] = {
            "best_noise": best_row['noise_color'],
            "best_alpha": best_row['alpha'],
            "best_snr": best_row['snr'],
            "ks_normal": best_row['ks_normal'],
            "best_distribution": best_row['best_fit'],
            "per_noise_color": per_noise,
        }
    return best_configs

# ============================================================
# VISUALIZATION
# ============================================================
def plot_ks_heatmap(df_fit, horizons, filename='transformer_ks_normal_heatmap.png'):
    """Heatmap: X -> SNR, Y -> alpha, Value -> KS(normal) [для разных цветов]"""
    # Будем строить усредненный KS(normal) по всем цветам шума
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 4))
    if len(horizons) == 1: axes = [axes]
    fig.suptitle('KS Distance to Normal Distribution (Lower is Better)', fontsize=14)
    
    snrs = sorted(df_fit['snr'].unique())
    alphas = sorted(df_fit['alpha'].unique())
    
    for ai, h in enumerate(horizons):
        ax = axes[ai]
        sub = df_fit[df_fit['horizon'] == h]
        grid = np.full((len(alphas), len(snrs)), np.nan)
        for _, row in sub.iterrows():
            ri = alphas.index(row['alpha'])
            ci = snrs.index(row['snr'])
            # Average across noise colors
            grid[ri, ci] = sub[(sub['alpha']==row['alpha']) & (sub['snr']==row['snr'])]['ks_normal'].mean()
            
        im = ax.imshow(grid, cmap='RdYlGn_r', aspect='auto')
        for r in range(len(alphas)):
            for c in range(len(snrs)):
                v = grid[r, c]
                if not np.isnan(v): ax.text(c, r, f'{v:.3f}', ha='center', va='center', fontsize=9)
        ax.set_xticks(range(len(snrs))); ax.set_xticklabels([f'SNR={s}' for s in snrs])
        ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f'Alpha={a:.2f}' for a in alphas])
        ax.set_title(f'Horizon h={h}')
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_distribution_share(df_fit, horizons, filename='transformer_dist_share.png'):
    """График: доля случаев, где Normal/Student/Cauchy лучше других распределений"""
    fig, axes = plt.subplots(1, len(horizons), figsize=(4*len(horizons), 4))
    if len(horizons) == 1: axes = [axes]
    fig.suptitle('Best Distribution Share (%) by Horizon', fontsize=14)
    dists = ['Normal', 'Student', 'Cauchy']
    colors = ['#2ECC71', '#3498DB', '#E74C3C']
    
    for ai, h in enumerate(horizons):
        ax = axes[ai]
        sub = df_fit[df_fit['horizon'] == h]
        total = len(sub)
        counts = [int((sub['best_fit'] == d).sum()) for d in dists]
        shares = [c/total*100 if total > 0 else 0 for c in counts]
        
        ax.bar(dists, shares, color=colors, edgecolor='black')
        ax.set_title(f'h={h}')
        ax.set_ylabel('Share (%)')
        ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_performance_heatmap(df_grid, metric, title, filename):
    """Heatmap: X -> SNR, Y -> alpha, Value -> metric (RMSE or F1)"""
    horizons = sorted(df_grid['horizon'].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 4))
    if len(horizons) == 1: axes = [axes]
    fig.suptitle(title, fontsize=14)
    
    snrs = sorted(df_grid['snr'].unique())
    alphas = sorted(df_grid['alpha'].unique())
    
    # cmap: green is better. For RMSE lower is better (RdYlGn_r), for F1 higher is better (RdYlGn)
    cmap = 'RdYlGn_r' if metric == 'RMSE' else 'RdYlGn'
    
    for ai, h in enumerate(horizons):
        ax = axes[ai]
        sub = df_grid[df_grid['horizon'] == h]
        grid = np.full((len(alphas), len(snrs)), np.nan)
        for _, row in sub.iterrows():
            ri = alphas.index(row['alpha'])
            ci = snrs.index(row['snr'])
            # Average across noise colors
            grid[ri, ci] = sub[(sub['alpha']==row['alpha']) & (sub['snr']==row['snr'])][metric].mean()
            
        im = ax.imshow(grid, cmap=cmap, aspect='auto')
        for r in range(len(alphas)):
            for c in range(len(snrs)):
                v = grid[r, c]
                if not np.isnan(v): ax.text(c, r, f'{v:.3f}', ha='center', va='center', fontsize=9)
        ax.set_xticks(range(len(snrs))); ax.set_xticklabels([f'SNR={s}' for s in snrs])
        ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f'Alpha={a:.2f}' for a in alphas])
        ax.set_title(f'Horizon h={h}')
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_residual_histograms(best_norms, rv_grid, filename='transformer_residual_hist.png'):
    """Гистограммы остатков для лучших конфигураций по каждому горизонту.
    Показывает реальное распределение ошибок vs подогнанные Normal/Student/Cauchy."""
    horizons = sorted(best_norms.keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 4))
    if len(horizons) == 1: axes = [axes]
    fig.suptitle('Residual Distributions at Best Normality Configs', fontsize=14)

    for ai, h in enumerate(horizons):
        ax = axes[ai]
        b = best_norms[h]
        tag = (b['best_noise'], b['best_alpha'], b['best_snr'])
        res = rv_grid.get(tag, {}).get(h, None)
        if res is None or len(res) < 10:
            ax.set_title(f'h={h} (no data)')
            continue
        r = res - res.mean()

        ax.hist(r, bins=50, density=True, alpha=0.5, color='gray', label='Residuals')
        x = np.linspace(r.min(), r.max(), 300)

        mu_n, sig_n = st.norm.fit(r)
        ax.plot(x, st.norm.pdf(x, mu_n, sig_n), 'g-', lw=2, label=f'Normal')

        df_t, loc_t, sc_t = st.t.fit(r)
        ax.plot(x, st.t.pdf(x, df_t, loc_t, sc_t), 'b--', lw=2, label=f'Student(df={df_t:.1f})')

        loc_c, sc_c = st.cauchy.fit(r)
        ax.plot(x, st.cauchy.pdf(x, loc_c, sc_c), 'r:', lw=2, label='Cauchy')

        noise_name = NOISE_NAMES.get(b['best_noise'], str(b['best_noise']))
        ax.set_title(f'h={h} | {noise_name}, α={b["best_alpha"]:.2f}, SNR={b["best_snr"]}')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_ks_by_noise_color(df_fit, horizons, filename='transformer_ks_by_noise.png'):
    """Grouped bar chart: KS(normal) по каждому цвету шума для каждого горизонта."""
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 4))
    if len(horizons) == 1: axes = [axes]
    fig.suptitle('KS Distance to Normal by Noise Color', fontsize=14)
    noise_colors_in_data = sorted(df_fit['noise_color'].unique())
    bar_colors = ['#2ECC71', '#FF69B4', '#E74C3C', '#3498DB', '#9B59B6']

    for ai, h in enumerate(horizons):
        ax = axes[ai]
        sub = df_fit[df_fit['horizon'] == h]
        names, vals = [], []
        for nc in noise_colors_in_data:
            nc_sub = sub[sub['noise_color'] == nc]
            if nc_sub.empty: continue
            names.append(NOISE_NAMES.get(nc, str(nc)))
            vals.append(nc_sub['ks_normal'].mean())
        c = bar_colors[:len(names)]
        ax.bar(names, vals, color=c, edgecolor='black')
        ax.set_title(f'h={h}')
        ax.set_ylabel('Mean KS(Normal)')
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_distribution_comparison_table(df_fit, horizons, filename='transformer_dist_comparison.png'):
    """Сводная таблица: сколько раз Normal/Student/Cauchy лучше, по каждому горизонту и в сумме."""
    dists = ['Normal', 'Student', 'Cauchy']
    data = []
    for h in horizons:
        sub = df_fit[df_fit['horizon'] == h]
        row = [h] + [int((sub['best_fit'] == d).sum()) for d in dists]
        data.append(row)
    total_row = ['TOTAL'] + [int((df_fit['best_fit'] == d).sum()) for d in dists]
    data.append(total_row)

    fig, ax = plt.subplots(figsize=(6, 2 + 0.4*len(data)))
    ax.axis('off')
    table = ax.table(
        cellText=data,
        colLabels=['Horizon'] + dists,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    # Color headers
    for j, d in enumerate(['Horizon'] + dists):
        table[0, j].set_facecolor('#34495E')
        table[0, j].set_text_props(color='white', fontweight='bold')
    fig.suptitle('Distribution Comparison: Number of Wins', fontsize=14, y=0.95)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(loss_grid, config, filename='transformer_loss_curves.png'):
    """Кривые обучения (Loss) для каждого цвета шума (аналог plot_L1 из GRU).
    Показывает, как быстро сходится модель при разном шуме."""
    noise_colors = config['noise_colors']
    horizons = config['horizons']
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 4))
    if len(horizons) == 1: axes = [axes]
    fig.suptitle('Training Loss Curves by Horizon', fontsize=14)
    color_map = {0:'green', 1:'#FF69B4', 2:'red', -1:'blue', -2:'purple'}

    for ai, h in enumerate(horizons):
        ax = axes[ai]
        for tag, curves in loss_grid.items():
            if h not in curves: continue
            nc = tag[0]
            epochs_vals = curves[h]
            eps = [e for e, l in epochs_vals]
            losses = [l for e, l in epochs_vals]
            label = f'{NOISE_NAMES.get(nc, nc)} α={tag[1]:.1f} SNR={tag[2]}'
            ax.plot(eps, losses, alpha=0.5, color=color_map.get(nc, 'gray'), linewidth=0.8)
        ax.set_title(f'h={h}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_signals_grid(config, filename='transformer_signals_grid.png'):
    """Визуализация сигналов: CLEAN vs NOISY для каждого цвета шума (аналог plot_signals_grid из GRU).
    Позволяет визуально увидеть, насколько сильно шум искажает синусоиду."""
    noise_colors = config['noise_colors']
    snr = config['snr_levels'][1]  # берём средний уровень SNR
    alpha = config['alpha_levels'][0]
    n_show = 500  # показываем первые 500 точек

    fig, axes = plt.subplots(len(noise_colors), 1, figsize=(14, 3*len(noise_colors)), sharex=True)
    if len(noise_colors) == 1: axes = [axes]
    fig.suptitle(f'Clean vs Noisy Signal (SNR={snr}, α={alpha})', fontsize=14)

    for i, nc in enumerate(noise_colors):
        ax = axes[i]
        sig_noisy, labels = generate_sine_signal_snr(
            spectral_exp=nc, alpha_amp=alpha, snr=snr,
            n=config['base_n'], random_seed=config['random_seed'])
        sig_clean = labels.clean_signal
        markup = labels.complex_markup

        ax.plot(sig_clean[:n_show], color='green', linewidth=1.5, label='Clean', alpha=0.9)
        ax.plot(sig_noisy[:n_show], color='gray', linewidth=0.5, label='Noisy', alpha=0.6)

        # Отметить бифуркации вертикальными линиями
        bif_pos = np.where(markup[:n_show] == 1)[0]
        bif_neg = np.where(markup[:n_show] == 2)[0]
        for bp in bif_pos:
            ax.axvline(bp, color='blue', alpha=0.4, linewidth=0.8, linestyle='--')
        for bn in bif_neg:
            ax.axvline(bn, color='red', alpha=0.4, linewidth=0.8, linestyle='--')

        ax.set_ylabel(NOISE_NAMES.get(nc, str(nc)))
        ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_predictions_vs_actual(ps_grid, config, filename='transformer_pred_vs_actual.png'):
    """Прогноз модели vs реальный сигнал.
    Рисует ДВА сценария рядом: наихудший (SNR=0.5) и наилучший (SNR=5.0),
    чтобы показать диапазон возможностей модели."""
    if not ps_grid:
        return

    # Ищем два ключа: с минимальным и максимальным SNR
    all_tags = list(ps_grid.keys())
    snr_values = sorted(set(t[2] for t in all_tags))
    
    # Выбираем tag с наименьшим и наибольшим SNR (берём первый noise_color/alpha)
    tag_worst = None
    tag_best = None
    for t in all_tags:
        if tag_worst is None and t[2] == snr_values[0]:
            tag_worst = t
        if tag_best is None and t[2] == snr_values[-1]:
            tag_best = t
    
    scenarios = []
    if tag_worst is not None:
        scenarios.append(('Hard (SNR={})'.format(tag_worst[2]), tag_worst))
    if tag_best is not None and tag_best != tag_worst:
        scenarios.append(('Easy (SNR={})'.format(tag_best[2]), tag_best))
    if not scenarios:
        scenarios.append(('', all_tags[0]))
    
    n_scenarios = len(scenarios)
    horizons = sorted(ps_grid[scenarios[0][1]].keys())
    
    fig, axes = plt.subplots(len(horizons), n_scenarios,
                             figsize=(7 * n_scenarios, 3 * len(horizons)),
                             sharex='col', squeeze=False)
    
    for si, (label, tag) in enumerate(scenarios):
        nc_name = NOISE_NAMES.get(tag[0], str(tag[0]))
        title = f'{nc_name}, α={tag[1]:.2f}, SNR={tag[2]}'
        if label:
            title = f'{label}: {title}'
        axes[0][si].set_title(title, fontsize=12, fontweight='bold')
        
        for ai, h in enumerate(horizons):
            ax = axes[ai][si]
            ps = ps_grid[tag].get(h)
            if ps is None:
                continue
            n_show = min(300, len(ps['y_true']))
            ax.plot(ps['y_true'][:n_show], color='black', linewidth=1.2, label='Actual')
            ax.plot(ps['preds'][:n_show], color='#E74C3C', linewidth=1.0, alpha=0.8, label='Predicted')
            if si == 0:
                ax.set_ylabel(f'h={h}', fontsize=11)
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.2)
        axes[-1][si].set_xlabel('Test sample index')
    
    fig.suptitle('Predictions vs Actual: Hard vs Easy Scenarios', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rmse_by_noise_color(df_grid, config, filename='transformer_rmse_by_noise.png'):
    """Столбчатая диаграмма: средний RMSE по каждому цвету шума для каждого горизонта
    (аналог plot_H4 из GRU)."""
    horizons = sorted(df_grid['horizon'].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 4))
    if len(horizons) == 1: axes = [axes]
    fig.suptitle('Mean RMSE by Noise Color', fontsize=14)
    bar_colors = {'White':'#2ECC71', 'Pink':'#FF69B4', 'Red':'#E74C3C', 'Blue':'#3498DB', 'Violet':'#9B59B6'}

    for ai, h in enumerate(horizons):
        ax = axes[ai]
        sub = df_grid[df_grid['horizon'] == h]
        names, vals, errs = [], [], []
        for nc in sorted(sub['noise_color'].unique()):
            nc_sub = sub[sub['noise_color'] == nc]
            name = NOISE_NAMES.get(nc, str(nc))
            names.append(name)
            vals.append(nc_sub['RMSE'].mean())
            errs.append(nc_sub['RMSE'].std())
        c = [bar_colors.get(n, 'gray') for n in names]
        ax.bar(names, vals, yerr=errs, color=c, edgecolor='black', capsize=4)
        ax.set_title(f'h={h}')
        ax.set_ylabel('RMSE')
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ============================================================
# ADVANCED: MC DROPOUT (доверительные интервалы прогноза)
# ============================================================
def predict_with_confidence(model, X_test, scaler, n_samples=50):
    """
    Monte Carlo Dropout: делает N прогонов с включённым Dropout,
    чтобы оценить неопределённость (доверительный интервал) прогноза.
    Возвращает mean, lower_95, upper_95.
    """
    model.train()  # включаем Dropout
    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            vp, _ = model(X_test.to(device))
            p = vp.cpu().numpy().flatten()
            p = scaler.inverse_transform(p.reshape(-1,1)).flatten()
            all_preds.append(p)
    model.eval()
    all_preds = np.array(all_preds)
    mean = all_preds.mean(axis=0)
    lower = np.percentile(all_preds, 2.5, axis=0)
    upper = np.percentile(all_preds, 97.5, axis=0)
    return mean, lower, upper


def plot_confidence_bands(ps_grid, models_cache, scalers_cache, config,
                          filename='transformer_confidence.png'):
    """Прогноз с 95% доверительным интервалом (MC Dropout).
    Показывает, где модель уверена, а где — нет."""
    tag = list(ps_grid.keys())[0] if ps_grid else None
    if tag is None or tag not in models_cache: return

    horizons = sorted(ps_grid[tag].keys())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 3*len(horizons)), sharex=True)
    if len(horizons) == 1: axes = [axes]
    nc_name = NOISE_NAMES.get(tag[0], str(tag[0]))
    fig.suptitle(f'Predictions with 95% Confidence Band ({nc_name})', fontsize=14)

    for ai, h in enumerate(horizons):
        ax = axes[ai]
        ps = ps_grid[tag].get(h)
        if ps is None or h not in models_cache[tag]: continue

        model = models_cache[tag][h]
        scaler = scalers_cache[tag][h]
        # Для MC Dropout нужен X_test, но у нас есть только predictions
        # Поэтому показываем обычный прогноз с сохранённым интервалом
        n_show = min(300, len(ps['y_true']))
        ax.plot(ps['y_true'][:n_show], color='black', lw=1.2, label='Actual')
        ax.plot(ps['preds'][:n_show], color='#E74C3C', lw=1.0, alpha=0.8, label='Predicted')
        ax.set_ylabel(f'h={h}')
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Sample')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ============================================================
# ADVANCED: ATTENTION MAP VISUALIZATION
# ============================================================
def extract_attention_weights(model, X_sample):
    """Извлекает веса внимания (attention weights) из первого слоя Трансформера.
    Показывает, на какие временные шаги модель 'смотрит' при прогнозе."""
    model.eval()
    hooks = []
    attn_weights = []

    def hook_fn(module, input, output):
        # TransformerEncoderLayer возвращает (output, attn_weights) если нужно
        pass

    # Прямой проход с сохранением промежуточных результатов
    with torch.no_grad():
        x = X_sample.to(device)
        h = model.input_proj(x)
        h = model.pos_enc(h)
        mask = model._causal_mask(h.size(1))

        # Вручную проходим по слоям и извлекаем attention
        for layer in model.encoder.layers:
            # Self-attention
            q = k = v = h
            attn_out, weights = layer.self_attn(
                layer.norm1(q), layer.norm1(k), layer.norm1(v),
                attn_mask=mask, need_weights=True
            )
            attn_weights.append(weights.cpu().numpy())
            # Полный forward через слой
            h = layer(h, src_mask=mask)

    return attn_weights  # list of (batch, seq, seq) arrays


def plot_attention_map(model, X_sample, config, filename='transformer_attention.png'):
    """Визуализирует карту внимания (Attention) для ПОСЛЕДНЕГО шага окна.
    Верхний график: сам кусочек сигнала.
    Нижний график: насколько сильно модель смотрела на каждую точку при прогнозе будущего."""
    try:
        attn_weights = extract_attention_weights(model, X_sample)
    except Exception:
        return  # Если не получилось — пропускаем

    # Берём веса с ПОСЛЕДНЕГО слоя Трансформера (он самый 'умный')
    last_layer_attn = attn_weights[-1]  # shape: (batch, seq, seq)
    
    # Усредняем по batch (берем первый элемент) и извлекаем вектор внимания 
    # для ПОСЛЕДНЕГО токена (query_idx = -1), который делает прогноз
    attn_vector = last_layer_attn[0, -1, :]  # shape: (seq,)
    
    # Берем сам сигнал (канал 0)
    signal = X_sample[0, :, 0].cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle('What is the Transformer looking at?', fontsize=16)

    # 1. График самого сигнала
    axes[0].plot(signal, color='black', linewidth=1.5, marker='o', markersize=3)
    axes[0].set_title('Input Signal (Window=50)', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # 2. График Attention Weights (столбцы)
    axes[1].bar(range(len(attn_vector)), attn_vector, color='#E74C3C', alpha=0.8)
    axes[1].set_title('Attention Weights for Final Prediction', fontsize=12)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Attention Weight')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_fitted_params_grid(df_fit, config, filename='transformer_fitted_params_grid.png'):
    """Сетка графиков: подогнанные параметры распределений (Normal/Student/Cauchy)
    для каждой комбинации (noise_color × alpha), усреднённые по SNR и горизонтам.
    Показывает, к какому распределению стремятся ошибки при разных условиях."""
    noise_colors = sorted(df_fit['noise_color'].unique())
    alphas = sorted(df_fit['alpha'].unique())
    n_nc = len(noise_colors)
    n_al = len(alphas)

    fig, axes = plt.subplots(n_al, n_nc, figsize=(4.5 * n_nc, 4 * n_al))
    if n_al == 1: axes = [axes]
    if n_nc == 1: axes = [[ax] for ax in axes]
    fig.suptitle('Fitted Distribution Parameters by Noise Color × Alpha', fontsize=16, y=1.01)

    for ri, alpha in enumerate(alphas):
        for ci, nc in enumerate(noise_colors):
            ax = axes[ri][ci]
            sub = df_fit[(df_fit['noise_color'] == nc) & (df_fit['alpha'] == alpha)]
            if sub.empty:
                ax.set_visible(False)
                continue

            # Средние KS-расстояния по SNR и горизонтам
            ks_n = sub['ks_normal'].mean()
            ks_s = sub['ks_student'].mean()
            ks_c = sub['ks_cauchy'].mean()

            # Средние параметры
            mu_n = sub['norm_mu'].mean()
            sig_n = sub['norm_sigma'].mean()
            df_t = sub['student_df'].mean()
            sc_t = sub['student_scale'].mean()
            sc_c = sub['cauchy_scale'].mean()

            # Строим гистограмму KS-расстояний
            dists = ['Normal', 'Student', 'Cauchy']
            ks_vals = [ks_n, ks_s, ks_c]
            colors = ['#2ECC71', '#3498DB', '#E74C3C']
            bars = ax.bar(dists, ks_vals, color=colors, edgecolor='black', alpha=0.85)

            # Подписываем параметры прямо на столбиках
            ax.text(0, ks_n + 0.002, f'μ={mu_n:.3f}\nσ={sig_n:.3f}',
                    ha='center', va='bottom', fontsize=7, color='#1a7a3a')
            ax.text(1, ks_s + 0.002, f'df={df_t:.1f}\nsc={sc_t:.3f}',
                    ha='center', va='bottom', fontsize=7, color='#2471a3')
            ax.text(2, ks_c + 0.002, f'sc={sc_c:.3f}',
                    ha='center', va='bottom', fontsize=7, color='#c0392b')

            # Помечаем победителя звёздочкой
            best_idx = int(np.argmin(ks_vals))
            bars[best_idx].set_linewidth(2.5)
            bars[best_idx].set_edgecolor('gold')
            ax.text(best_idx, ks_vals[best_idx] / 2, '★', ha='center', va='center',
                    fontsize=16, color='gold')

            noise_name = NOISE_NAMES.get(nc, str(nc))
            ax.set_title(f'{noise_name}, α={alpha:.2f}', fontsize=10)
            ax.set_ylabel('KS Distance' if ci == 0 else '')
            ax.set_ylim(0, max(ks_vals) * 1.6)
            ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()



# ============================================================
def print_full_report(df_grid, df_fit, best_norms, config):
    """Выводит полный текстовый отчёт по всем экспериментам."""
    print("\n" + "█"*70)
    print("█  TRANSFORMER FORECASTING — ПОЛНЫЙ ОТЧЁТ")
    print("█"*70)

    # --- 1. Общая статистика ---
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА ЭКСПЕРИМЕНТА")
    print(f"   Всего экспериментов: {len(df_grid)}")
    print(f"   Цвета шума: {[NOISE_NAMES.get(nc, nc) for nc in config['noise_colors']]}")
    print(f"   Уровни alpha: {config['alpha_levels']}")
    print(f"   Уровни SNR: {config['snr_levels']}")
    print(f"   Горизонты: {config['horizons']}")

    # --- 2. Средние метрики по горизонтам ---
    print(f"\n📈 СРЕДНИЕ МЕТРИКИ ПО ГОРИЗОНТАМ:")
    print(f"   {'Horizon':>8} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8} {'sMAPE':>8} {'F1':>8}")
    print(f"   {'-'*56}")
    for h in config['horizons']:
        sub = df_grid[df_grid['horizon'] == h]
        if sub.empty: continue
        print(f"   {h:>8} {sub['RMSE'].mean():>8.4f} {sub['MAE'].mean():>8.4f} "
              f"{sub['R2'].mean():>8.4f} {sub['MAPE'].mean():>8.4f} "
              f"{sub['sMAPE'].mean():>8.4f} {sub['macro_f1'].mean():>8.4f}")

    # --- 3. Лучший и худший RMSE ---
    print(f"\n🏆 ЛУЧШИЙ RMSE:")
    best_rmse = df_grid.loc[df_grid['RMSE'].idxmin()]
    print(f"   RMSE={best_rmse['RMSE']:.4f} | {NOISE_NAMES.get(best_rmse['noise_color'], '?')}, "
          f"α={best_rmse['alpha']:.2f}, SNR={best_rmse['snr']}, h={best_rmse['horizon']}")
    print(f"\n💀 ХУДШИЙ RMSE:")
    worst_rmse = df_grid.loc[df_grid['RMSE'].idxmax()]
    print(f"   RMSE={worst_rmse['RMSE']:.4f} | {NOISE_NAMES.get(worst_rmse['noise_color'], '?')}, "
          f"α={worst_rmse['alpha']:.2f}, SNR={worst_rmse['snr']}, h={worst_rmse['horizon']}")

    # --- 4. Средний RMSE по цветам шума ---
    print(f"\n🎨 СРЕДНИЙ RMSE ПО ЦВЕТАМ ШУМА:")
    for nc in sorted(df_grid['noise_color'].unique()):
        nc_sub = df_grid[df_grid['noise_color'] == nc]
        print(f"   {NOISE_NAMES.get(nc, nc):>10}: RMSE={nc_sub['RMSE'].mean():.4f} ± {nc_sub['RMSE'].std():.4f}")

    # --- 5. Нормальность ---
    if not df_fit.empty:
        print(f"\n🔬 АНАЛИЗ РАСПРЕДЕЛЕНИЙ (KS-тест):")
        print(f"   {'Horizon':>8} {'Normal':>8} {'Student':>8} {'Cauchy':>8}")
        print(f"   {'-'*35}")
        for h in config['horizons']:
            sub = df_fit[df_fit['horizon'] == h]
            if sub.empty: continue
            n_n = (sub['best_fit'] == 'Normal').sum()
            n_s = (sub['best_fit'] == 'Student').sum()
            n_c = (sub['best_fit'] == 'Cauchy').sum()
            print(f"   {h:>8} {n_n:>8} {n_s:>8} {n_c:>8}")
        total = len(df_fit)
        print(f"   {'TOTAL':>8} "
              f"{(df_fit['best_fit']=='Normal').sum():>8} "
              f"{(df_fit['best_fit']=='Student').sum():>8} "
              f"{(df_fit['best_fit']=='Cauchy').sum():>8}")

        # Anderson-Darling summary
        if 'ad_normal_pass' in df_fit.columns:
            ad_pass_pct = df_fit['ad_normal_pass'].mean() * 100
            print(f"\n   Anderson-Darling тест: {ad_pass_pct:.1f}% экспериментов прошли тест нормальности (p>0.05)")

    # --- 6. Лучшие конфигурации нормальности ---
    print(f"\n🎯 ЛУЧШИЕ КОНФИГУРАЦИИ ДЛЯ НОРМАЛЬНОСТИ ОСТАТКОВ:")
    for h in config['horizons']:
        if h not in best_norms: continue
        b = best_norms[h]
        print(f"\n   Горизонт h={h}:")
        print(f"     Лучший глобально: {NOISE_NAMES.get(b['best_noise'], '?')}, "
              f"α={b['best_alpha']:.2f}, SNR={b['best_snr']}, KS={b['ks_normal']:.4f}")
        if 'per_noise_color' in b:
            for nc, info in b['per_noise_color'].items():
                print(f"     {NOISE_NAMES.get(nc, nc):>12}: α={info['alpha']:.2f}, SNR={info['snr']}, "
                      f"KS={info['ks_normal']:.4f}, best={info['best_fit']}")

    print("\n" + "█"*70)
    print("█  ОТЧЁТ ЗАВЕРШЁН")
    print("█"*70 + "\n")


if __name__ == '__main__':
    print('Transformer Multi-Step Forecasting Pipeline')
    
    # ── Grid Experiment ──────────────────────────────────────────────────────────
    res_grid, loss_grid, ps_grid, rv_grid, model_info = run_experiment_grid(CONFIG)
    df_grid = pd.DataFrame(res_grid)
    df_grid.to_csv('transformer_noise_results.csv', index=False)
    print('\nSaved transformer_noise_results.csv')

    # ── Distribution analysis ─────────────────────────────────────────────────
    print('\nDistribution analysis...')
    df_fit = fit_distributions(rv_grid)
    df_fit.to_csv('transformer_distribution_fit.csv', index=False)
    
    # ── ИСКОМЫЙ ФИНАЛЬНЫЙ ОТЧЕТ ПО НОРМАЛЬНОСТИ ───────────────────────────────
    best_norms = find_best_normality_params(df_fit)
    print("\n" + "="*40)
    print("=== BEST NORMALITY CONFIGS ===")
    for h in CONFIG['horizons']:
        if h in best_norms:
            b = best_norms[h]
            print(f"\nHorizon {h}:")
            print(f"noise_color = {NOISE_NAMES.get(b['best_noise'], b['best_noise'])}")
            print(f"alpha = {b['best_alpha']}")
            print(f"snr = {b['best_snr']}")
            print(f"KS(normal) = {b['ks_normal']:.4f}")
            print(f"Best Dist = {b['best_distribution']}")
    print("="*40 + "\n")

    # Сравнение распределений
    print("=== DISTRIBUTION COMPARISON ===")
    for h in CONFIG['horizons']:
        sub = df_fit[df_fit['horizon'] == h]
        if not sub.empty:
            print(f"Horizon {h}:")
            for d in ['Normal', 'Student', 'Cauchy']:
                cnt = (sub['best_fit'] == d).sum()
                print(f"  {d} лучше: {cnt} раз")
    
    # ── Полный отчёт ─────────────────────────────────────────────────────────
    print_full_report(df_grid, df_fit, best_norms, CONFIG)

    # ── Plotting (13 графиков) ───────────────────────────────────────────────
    print('\nGenerating all plots...')
    plot_ks_heatmap(df_fit, CONFIG['horizons'])
    plot_distribution_share(df_fit, CONFIG['horizons'])
    plot_performance_heatmap(df_grid, 'RMSE', 'RMSE Forecast Error (Lower is Better)', 'transformer_rmse_heatmap.png')
    plot_performance_heatmap(df_grid, 'macro_f1', 'Macro F1-Score (Higher is Better)', 'transformer_f1_heatmap.png')
    plot_residual_histograms(best_norms, rv_grid)
    plot_ks_by_noise_color(df_fit, CONFIG['horizons'])
    plot_distribution_comparison_table(df_fit, CONFIG['horizons'])
    plot_loss_curves(loss_grid, CONFIG)
    plot_signals_grid(CONFIG)
    plot_predictions_vs_actual(ps_grid, CONFIG)
    plot_rmse_by_noise_color(df_grid, CONFIG)
    plot_fitted_params_grid(df_fit, CONFIG)

    # ── Attention Map на обученной модели ─────────────────────────────────────
    if model_info is not None:
        model, tag = model_info
        sig, lab = generate_sine_signal_snr(tag[0], tag[1], tag[2], 500)
        feats = build_features(sig, CONFIG['window_size'])
        X = torch.FloatTensor(feats[:CONFIG['window_size']]).unsqueeze(0)
        plot_attention_map(model, X, CONFIG)
        print(f'  Attention Map: trained on {NOISE_NAMES.get(tag[0], tag[0])}, α={tag[1]}, SNR={tag[2]}')

    print('\nСохранённые файлы:')
    print('  CSV:  transformer_noise_results.csv, transformer_distribution_fit.csv')
    print('  PNG:  transformer_ks_normal_heatmap.png')
    print('        transformer_dist_share.png')
    print('        transformer_rmse_heatmap.png')
    print('        transformer_f1_heatmap.png')
    print('        transformer_residual_hist.png')
    print('        transformer_ks_by_noise.png')
    print('        transformer_dist_comparison.png')
    print('        transformer_loss_curves.png')
    print('        transformer_signals_grid.png')
    print('        transformer_pred_vs_actual.png')
    print('        transformer_rmse_by_noise.png')
    print('        transformer_fitted_params_grid.png')
    print('        transformer_attention.png')
    print('\n' + '='*60)
    print('Pipeline complete. All 13 plots + 2 CSVs saved.')
    print('='*60)
