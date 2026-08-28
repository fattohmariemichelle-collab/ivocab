from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nq_trend_detector_v8_real_nq as v8
import nq_trend_detector_v8b_real_nq as v8b
import nq_trend_detector_v8c_ablation as v8c

OUT = Path('trend_backtest_v8d_results')
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260828)

# Preserve corrected causal V8b methodology.
v8.session_features = v8b.causal_session_features
v8.add_labels = v8b.gap_safe_labels


def build_dataset():
    m = v8.load()
    quality = {
        'rows_1m': int(len(m)),
        'start': str(m.ts.min()),
        'end': str(m.ts.max()),
        'duplicates': int(m.ts.duplicated().sum()),
        'zero_volume': int((m.volume <= 0).sum()),
    }
    m = v8.session_features(m)
    h = v8.make_h1(m)
    d = v8.make_context(m, h)
    d = v8.add_h4_d1(d, m)
    d = v8.add_labels(d, 12)

    # Explicit signed directional features. All are known at the H1 close.
    d['ema20_50_signed_atr'] = (d.ema20 - d.ema50) / d.atr.replace(0, np.nan)
    d['ema10_20_signed_atr'] = (d.ema10 - d.ema20) / d.atr.replace(0, np.nan)
    d['dmi_signed'] = (d.pdi - d.mdi) / 100.0
    d['close_ema20_signed_atr'] = (d.close - d.ema20) / d.atr.replace(0, np.nan)
    d['h4_ema_signed'] = np.sign(d.h4_ema20 - d.h4_ema50)
    d['d1_ema_signed'] = np.sign(d.d1_ema20 - d.d1_ema50)
    return d, quality


def feature_sets():
    price_dir = [
        'ema20_50_signed_atr','ema10_20_signed_atr','dmi_signed','close_ema20_signed_atr',
        'mom6_atr','mom12_atr','adx','er10','er20','rv6','rv24',
        'h4_ema_signed','h4_dir','h4_adx','h4_er10',
        'd1_ema_signed','d1_dir','d1_adx','d1_er10',
    ]
    location = [
        'dist_vwap_atr','dist_prev_high_atr','dist_prev_low_atr',
        'dist_on_high_atr','dist_on_low_atr','dist_or_high_atr','dist_or_low_atr',
        'on_range_atr','or_range_atr',
    ]
    volume = ['rel_vol24','on_vol_ratio','or_vol_ratio']
    session = ['hour','dow','is_rth','is_opening_hour','is_power_hour','is_overnight']
    sets = {
        'price_direction': price_dir,
        'price_plus_location': price_dir + location,
        'price_plus_location_volume': price_dir + location + volume,
        'full_direction': price_dir + location + volume + session,
    }
    return sets


def trend_gate_features():
    # V8c's simpler near-best trend engine: technical price + causal overnight/opening block.
    sets, _ = v8c.feature_sets()
    return sets['price_plus_overnight_opening']


def model_candidates():
    return {
        'logit': make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced', random_state=42),
        ),
        'hgb': make_pipeline(
            SimpleImputer(strategy='median'),
            HistGradientBoostingClassifier(
                max_iter=180, max_depth=3, learning_rate=0.035,
                l2_regularization=1.5, random_state=42,
            ),
        ),
    }


def safe_auc(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def direction_metrics(y, p, pred=None):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    if pred is None:
        pred = (p >= 0.5).astype(int)
    else:
        pred = np.asarray(pred, dtype=int)
    if len(y) == 0:
        return {'n': 0}
    cm = confusion_matrix(y, pred, labels=[0,1])
    return {
        'n': int(len(y)),
        'long_base_rate': float(y.mean()),
        'auc': safe_auc(y, p),
        'ap_long': float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        'brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, np.column_stack([1-p,p]), labels=[0,1])),
        'precision_long': float(precision_score(y, pred, pos_label=1, zero_division=0)),
        'recall_long': float(recall_score(y, pred, pos_label=1, zero_division=0)),
        'precision_short': float(precision_score(y, pred, pos_label=0, zero_division=0)),
        'recall_short': float(recall_score(y, pred, pos_label=0, zero_division=0)),
        'confusion_short_long': cm.tolist(),
    }


def abstention_metrics(y, p, margin):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    keep = np.abs(p - 0.5) >= margin
    pred = (p[keep] >= 0.5).astype(int)
    r = direction_metrics(y[keep], p[keep], pred)
    r['coverage'] = float(keep.mean()) if len(keep) else 0.0
    r['abstention_margin'] = float(margin)
    return r, keep


def bootstrap_accuracy_balanced(y, pred, reps=3000):
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=int)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {'accuracy_ci95': [np.nan,np.nan], 'balanced_accuracy_ci95': [np.nan,np.nan]}
    acc = []; bal = []
    n = len(y)
    for _ in range(reps):
        idx = RNG.integers(0, n, n)
        yy, pp = y[idx], pred[idx]
        if len(np.unique(yy)) < 2:
            continue
        acc.append(accuracy_score(yy, pp)); bal.append(balanced_accuracy_score(yy, pp))
    return {
        'accuracy_ci95': [float(np.quantile(acc,.025)), float(np.quantile(acc,.975))],
        'balanced_accuracy_ci95': [float(np.quantile(bal,.025)), float(np.quantile(bal,.975))],
    }


def calibration_table(y, p, bins=5):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return []
    edges = np.linspace(0,1,bins+1)
    rows=[]
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]
        mask=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if mask.any():
            rows.append({'p_low':float(lo),'p_high':float(hi),'n':int(mask.sum()),'mean_pred_long':float(p[mask].mean()),'actual_long_rate':float(y[mask].mean())})
    return rows


def fit_trend_gate(d):
    feats = trend_gate_features()
    tr = d[(d.ts >= '2023-01-01') & (d.ts < '2024-01-01')].dropna(subset=['trend_clean'])
    va = d[(d.ts >= '2024-01-01') & (d.ts < '2025-01-01')].dropna(subset=['trend_clean'])
    models = model_candidates()
    cand=[]
    for name, model in models.items():
        model.fit(tr[feats], tr.trend_clean.astype(int))
        pv=model.predict_proba(va[feats])[:,1]
        cand.append((average_precision_score(va.trend_clean.astype(int),pv),name,model,pv))
    cand.sort(key=lambda z:z[0], reverse=True)
    ap,name,model,pv=cand[0]
    gate_cut=float(np.quantile(pv,0.90))
    return {'model':name,'validation_ap':float(ap),'validation_gate_cut':gate_cut,'features':feats}, model


def prepare_direction_period(d, trend_model, trend_info, start, end):
    x=d[(d.ts>=start)&(d.ts<end)].dropna(subset=['trend_clean']).copy()
    x['trend_prob']=trend_model.predict_proba(x[trend_info['features']])[:,1]
    x['gate']=x.trend_prob>=trend_info['validation_gate_cut']
    # Direction is scored only where a clean trend actually followed. This measures conditional direction skill.
    q=x[(x.gate)&(x.trend_clean==1)].copy()
    q['y_long']=(q.fwd_atr>0).astype(int)
    return x,q


def choose_direction_model(d, trend_model, trend_info, features):
    tr=d[(d.ts>='2023-01-01')&(d.ts<'2024-01-01')&(d.trend_clean==1)].copy()
    tr['y_long']=(tr.fwd_atr>0).astype(int)
    _,va=prepare_direction_period(d,trend_model,trend_info,'2024-01-01','2025-01-01')
    cand=[]
    for name,model in model_candidates().items():
        model.fit(tr[features],tr.y_long)
        pv=model.predict_proba(va[features])[:,1]
        m=direction_metrics(va.y_long,pv)
        # Selection is validation-only. Primary: balanced accuracy; tie-break: AUC then lower Brier.
        score=(m.get('balanced_accuracy',-1),m.get('auc',-1),-m.get('brier',99))
        cand.append((score,name,model,m,pv))
    cand.sort(key=lambda z:z[0],reverse=True)
    _,name,model,val_metrics,pv=cand[0]

    # Prespecified abstention grid. Require >=35% coverage and >=40 validation predictions.
    options=[]
    for margin in [0.00,0.05,0.10,0.15,0.20,0.25,0.30]:
        m,keep=abstention_metrics(va.y_long.to_numpy(),pv,margin)
        eligible=m.get('n',0)>=40 and m.get('coverage',0)>=0.35 and np.isfinite(m.get('balanced_accuracy',np.nan))
        options.append({'margin':margin,'eligible':bool(eligible),**m})
    eligible=[o for o in options if o['eligible']]
    if eligible:
        eligible.sort(key=lambda o:(o['balanced_accuracy'],o['accuracy'],o['coverage']),reverse=True)
        chosen_margin=float(eligible[0]['margin'])
    else:
        chosen_margin=0.0
    return name,model,val_metrics,options,chosen_margin,va,pv


def benchmarks(q):
    y=q.y_long.to_numpy(int)
    rows=[]
    def add(name,pred,mask=None):
        if mask is None: mask=np.ones(len(q),dtype=bool)
        mask=np.asarray(mask,dtype=bool); pp=np.asarray(pred,dtype=int)
        if not mask.any(): return
        rows.append({'rule':name,'coverage':float(mask.mean()),**direction_metrics(y[mask],pp[mask].astype(float),pp[mask])})
    add('always_long',np.ones(len(q),dtype=int))
    h1=np.sign(q.ema20-q.ema50).to_numpy(int); add('h1_ema', (h1>0).astype(int), h1!=0)
    h4=q.h4_dir.to_numpy(int); align=(h1==h4)&(h1!=0); add('h1_h4_ema_aligned',(h1>0).astype(int),align)
    dmi=np.sign(q.pdi-q.mdi).to_numpy(int); add('dmi',(dmi>0).astype(int),dmi!=0)
    vw=np.sign(q.close-q.session_vwap).to_numpy(int); add('vwap_side',(vw>0).astype(int),vw!=0)
    return rows


def evaluate_feature_set(d,trend_model,trend_info,name,features):
    model_name,model,val_metrics,margin_grid,margin,va,pv=choose_direction_model(d,trend_model,trend_info,features)
    test_all,test=prepare_direction_period(d,trend_model,trend_info,'2025-01-01','2025-12-01')
    pt=model.predict_proba(test[features])[:,1]
    test_metrics=direction_metrics(test.y_long,pt)
    abst,keep=abstention_metrics(test.y_long.to_numpy(),pt,margin)
    pred=(pt[keep]>=0.5).astype(int)
   