"""
╔══════════════════════════════════════════════════════════════════════╗
║  वृद्धि — VRIDDHI                                                   ║
║  NCD + Equity Reinvestment Intelligence System                      ║
║  Hemrek Capital                                                      ║
║                                                                      ║
║  Sanskrit: वृद्धि (Vriddhi) — "growth, increase, interest earned    ║
║  on capital." In classical financial Sanskrit, vriddhi specifically   ║
║  denotes the interest or yield that capital generates — making it    ║
║  the precise term for a system that converts debt yield into         ║
║  equity wealth.                                                      ║
║                                                                      ║
║  Design: Hemrek Capital Dark Theme (Nirnay/Pragyam lineage)         ║
║  Engine: Adam Deep Literature Mode                                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# INDIAN NUMBER FORMATTING
# ═══════════════════════════════════════════════════════════════

def _indian_commas(n: int) -> str:
    """Place commas Indian-style: 1,00,00,000 not 10,000,000."""
    s = str(abs(int(n)))
    if len(s) <= 3:
        return s
    result = s[-3:]
    s = s[:-3]
    while s:
        chunk = s[-2:] if len(s) >= 2 else s
        result = chunk + ',' + result
        s = s[:-2]
    return result


def fmt_inr(value: float, show_paise: bool = False) -> str:
    """Format as ₹10,00,000 or ₹10,00,000.50"""
    sign = '-' if value < 0 else ''
    abs_val = abs(value)
    if show_paise:
        integer_part = int(abs_val)
        decimal_part = round((abs_val - integer_part) * 100)
        return f'{sign}₹{_indian_commas(integer_part)}.{decimal_part:02d}'
    return f'{sign}₹{_indian_commas(round(abs_val))}'


def fmt_lakhs(value: float, decimals: int = 2) -> str:
    """Format as ₹10.00L"""
    return f'₹{value / 1e5:.{decimals}f}L'


def fmt_crores(value: float, decimals: int = 2) -> str:
    """Format as ₹1.50Cr"""
    return f'₹{value / 1e7:.{decimals}f}Cr'


def fmt_smart(value: float) -> str:
    """Auto-select L or Cr based on magnitude."""
    abs_val = abs(value)
    if abs_val >= 1e7:
        return fmt_crores(value)
    if abs_val >= 1e5:
        return fmt_lakhs(value)
    return fmt_inr(value)


def fmt_pct(value: float, decimals: int = 2) -> str:
    return f'{value * 100:.{decimals}f}%'


def fmt_x(value: float) -> str:
    return f'{value:.2f}x'


# ═══════════════════════════════════════════════════════════════
# HEMREK CAPITAL DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════

C = {
    'bg': '#0A0E17', 'bg2': '#111827', 'card': '#1A1F2E',
    'input': '#0D1117', 'gold': '#FFC300', 'gold_dim': '#B8960F',
    'gold_bg': 'rgba(255,195,0,0.08)', 'gold_border': 'rgba(255,195,0,0.25)',
    'text': '#E8EAED', 'text2': '#9AA0A6', 'muted': '#5F6368',
    'green': '#00E676', 'red': '#FF5252', 'blue': '#448AFF',
    'cyan': '#00BCD4', 'purple': '#B388FF', 'orange': '#FF9100',
    'border': 'rgba(255,255,255,0.06)',
}


def inject_css():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp {{ background:{C['bg']}; color:{C['text']}; font-family:'Inter',sans-serif; }}
    .stApp > header {{ background:transparent!important; }}
    section[data-testid="stSidebar"] {{ background:{C['bg2']}!important; border-right:1px solid {C['gold_border']}!important; }}
    section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label {{ color:{C['text2']}!important; font-size:0.85rem!important; }}
    h1,h2,h3,h4 {{ font-family:'Inter',sans-serif!important; letter-spacing:-0.02em; }}
    h1 {{ font-weight:800!important; font-size:1.7rem!important; color:{C['text']}!important; }}
    h2 {{ font-weight:700!important; font-size:1.25rem!important; color:{C['gold']}!important; }}
    h3 {{ font-weight:600!important; font-size:1.05rem!important; color:{C['text']}!important; }}
    [data-testid="stMetric"] {{ background:{C['card']}; border:1px solid {C['gold_border']}; border-radius:8px; padding:14px 16px; }}
    [data-testid="stMetric"]:hover {{ border-color:{C['gold']}; box-shadow:0 0 15px rgba(255,195,0,0.08); }}
    [data-testid="stMetricLabel"] p {{ color:{C['muted']}!important; font-family:'Inter'!important; font-size:0.7rem!important; font-weight:600!important; text-transform:uppercase; letter-spacing:0.08em; }}
    [data-testid="stMetricValue"] {{ color:{C['gold']}!important; font-family:'JetBrains Mono',monospace!important; font-weight:600!important; font-size:1.25rem!important; }}
    [data-testid="stMetricDelta"] {{ font-family:'JetBrains Mono',monospace!important; font-size:0.75rem!important; }}
    .stNumberInput input,.stTextInput input {{ background:{C['input']}!important; color:{C['gold']}!important; border:1px solid {C['gold_border']}!important; border-radius:6px!important; font-family:'JetBrains Mono',monospace!important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:2px; background:{C['bg2']}; border-radius:8px; padding:3px; }}
    .stTabs [data-baseweb="tab"] {{ background:transparent; color:{C['muted']}; border-radius:6px; font-family:'Inter'; font-weight:600; font-size:0.8rem; }}
    .stTabs [aria-selected="true"] {{ background:{C['card']}!important; color:{C['gold']}!important; border-bottom:2px solid {C['gold']}!important; }}
    .streamlit-expanderHeader {{ background:{C['card']}!important; border:1px solid {C['border']}!important; border-radius:6px!important; color:{C['text2']}!important; font-family:'JetBrains Mono',monospace!important; font-size:0.85rem!important; }}
    .streamlit-expanderContent {{ background:{C['card']}!important; border:1px solid {C['border']}!important; border-top:none!important; }}
    .hdr {{ background:linear-gradient(135deg,{C['bg2']},{C['card']}); border:1px solid {C['gold_border']}; border-left:3px solid {C['gold']}; border-radius:8px; padding:18px 22px; margin-bottom:20px; }}
    .hdr h1 {{ margin:0!important; padding:0!important; font-size:1.5rem!important; }}
    .hdr .sub {{ color:{C['muted']}; font-family:'JetBrains Mono',monospace; font-size:0.72rem; margin-top:5px; letter-spacing:0.05em; }}
    .adam {{ background:{C['card']}; border:1px solid {C['border']}; border-left:3px solid {C['purple']}; border-radius:6px; padding:14px 18px; margin:10px 0; font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:{C['text2']}; line-height:1.65; }}
    .adam .tag {{ color:{C['purple']}; font-weight:700; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px; display:block; }}
    .adam code {{ color:{C['gold']}; background:rgba(255,195,0,0.08); padding:1px 5px; border-radius:3px; }}
    .thesis {{ background:linear-gradient(135deg,rgba(255,195,0,0.04),rgba(255,195,0,0.01)); border:1px solid {C['gold_border']}; border-radius:8px; padding:18px 22px; margin:12px 0; color:{C['text2']}; font-size:0.86rem; line-height:1.7; }}
    .thesis strong {{ color:{C['gold']}; }}
    .comp-win {{ color:{C['green']}; font-weight:700; }}
    .comp-lose {{ color:{C['red']}; font-weight:700; }}
    .divider {{ height:1px; background:linear-gradient(to right,transparent,{C['gold_border']},transparent); margin:24px 0; }}
    .risk-card {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:8px; padding:14px 18px; margin:6px 0; }}
    .risk-card .rt {{ font-weight:700; font-size:0.88rem; margin-bottom:3px; }}
    .risk-card .rb {{ color:{C['text2']}; font-size:0.8rem; line-height:1.5; }}
    </style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SYSTEM PARAMETERS
# ═══════════════════════════════════════════════════════════════

@dataclass
class Params:
    principal: float = 10_00_000
    ncd_rate: float = 0.118
    horizon: int = 10
    eq_return: float = 0.13
    eq_vol: float = 0.18
    expense_ratio: float = 0.005
    tax_slab: float = 0.30
    cess: float = 0.04
    tds_rate: float = 0.10
    ltcg_rate: float = 0.125
    stcg_rate: float = 0.20
    ltcg_exempt: float = 1_25_000
    surcharge: float = 0.0
    inflation: float = 0.06
    fd_rate: float = 0.07
    savings_rate: float = 0.04
    mc_paths: int = 5000

    @property
    def eff_tax(self) -> float:
        return self.tax_slab * (1 + self.cess) * (1 + self.surcharge)

    @property
    def gross_monthly(self) -> float:
        return self.principal * self.ncd_rate / 12

    @property
    def net_monthly(self) -> float:
        return self.gross_monthly * (1 - self.tds_rate)

    @property
    def net_eq_return(self) -> float:
        return self.eq_return - self.expense_ratio

    @property
    def monthly_eq_r(self) -> float:
        return (1 + self.net_eq_return) ** (1 / 12) - 1

    @property
    def N(self) -> int:
        return self.horizon * 12


# ═══════════════════════════════════════════════════════════════
# ANALYTICAL ENGINE
# ═══════════════════════════════════════════════════════════════

class Engine:
    """Core computational engine — all metric implementations live here."""

    def __init__(self, p: Params):
        self.p = p

    # ── Deterministic Simulations ─────────────────────────────

    def simulate_ncd_system(self, eq_return: Optional[float] = None) -> pd.DataFrame:
        """Month-by-month NCD+Equity system. Recurrence: V(n) = V(n-1)·(1+r) + S"""
        p = self.p
        er = eq_return if eq_return is not None else p.net_eq_return
        r_m = (1 + er) ** (1 / 12) - 1

        rows = []
        mf = cum_inv = cum_tax_int = 0.0
        for m in range(1, p.N + 1):
            gross = p.gross_monthly
            tds = gross * p.tds_rate
            net_recv = gross - tds
            tax_full = gross * p.eff_tax

            mf = mf * (1 + r_m) + net_recv
            cum_inv += net_recv
            cum_tax_int += tax_full
            gains = mf - cum_inv
            ltcg = max(0, (gains - p.ltcg_exempt)) * p.ltcg_rate

            rows.append({
                'month': m, 'year': (m - 1) // 12 + 1,
                'gross_interest': gross, 'tds': tds, 'net_received': net_recv,
                'tax_on_interest': tax_full,
                'sip_amount': net_recv, 'cum_invested': cum_inv,
                'mf_value': mf, 'mf_gains': gains,
                'ncd_principal': p.principal,
                'total_wealth': p.principal + mf,
                'cum_tax_interest': cum_tax_int,
                'ltcg_if_exit': ltcg,
                'total_tax_if_exit': cum_tax_int + ltcg,
                'net_wealth': p.principal + mf - cum_tax_int - ltcg,
            })
        return pd.DataFrame(rows)

    def simulate_pure_sip(self, eq_return: Optional[float] = None) -> pd.DataFrame:
        """Pure SIP — same monthly amount, no NCD principal preservation."""
        p = self.p
        er = eq_return if eq_return is not None else p.net_eq_return
        r_m = (1 + er) ** (1 / 12) - 1
        sip = p.net_monthly

        rows = []
        mf = cum_inv = 0.0
        for m in range(1, p.N + 1):
            mf = mf * (1 + r_m) + sip
            cum_inv += sip
            gains = mf - cum_inv
            ltcg = max(0, (gains - p.ltcg_exempt)) * p.ltcg_rate

            rows.append({
                'month': m, 'year': (m - 1) // 12 + 1,
                'sip_amount': sip, 'cum_invested': cum_inv,
                'mf_value': mf, 'mf_gains': gains,
                'ltcg_if_exit': ltcg,
                'net_value': mf - ltcg,
            })
        return pd.DataFrame(rows)

    def simulate_fd_reinvest(self) -> pd.DataFrame:
        """Alternative: FD interest reinvested into equity (lower yield, same structure)."""
        p = self.p
        fd_monthly_gross = p.principal * p.fd_rate / 12
        fd_monthly_net = fd_monthly_gross * (1 - p.tds_rate)
        r_m = p.monthly_eq_r

        rows = []
        mf = cum_inv = cum_tax = 0.0
        for m in range(1, p.N + 1):
            tax_full = fd_monthly_gross * p.eff_tax
            mf = mf * (1 + r_m) + fd_monthly_net
            cum_inv += fd_monthly_net
            cum_tax += tax_full
            gains = mf - cum_inv
            ltcg = max(0, (gains - p.ltcg_exempt)) * p.ltcg_rate
            rows.append({
                'month': m, 'year': (m - 1) // 12 + 1,
                'sip_amount': fd_monthly_net,
                'cum_invested': cum_inv, 'mf_value': mf, 'mf_gains': gains,
                'total_wealth': p.principal + mf,
                'cum_tax': cum_tax, 'ltcg_if_exit': ltcg,
                'net_wealth': p.principal + mf - cum_tax - ltcg,
            })
        return pd.DataFrame(rows)

    # ── Closed-Form Validation ────────────────────────────────

    def closed_form(self, sip: Optional[float] = None, eq_return: Optional[float] = None) -> float:
        """V(N) = S · [(1+r)^N - 1] / r  (ordinary annuity)"""
        p = self.p
        S = sip if sip is not None else p.net_monthly
        er = eq_return if eq_return is not None else p.net_eq_return
        r_m = (1 + er) ** (1 / 12) - 1
        if r_m == 0:
            return S * p.N
        return S * ((1 + r_m) ** p.N - 1) / r_m

    # ── Monte Carlo ───────────────────────────────────────────

    def monte_carlo(self) -> dict:
        """
        GBM Monte Carlo with drift correction for Jensen's inequality.
        μ_adj = ln(1+r)/12 - σ²/24 ensures E[multiplicative return] is unbiased.
        """
        p = self.p
        mu_m = np.log(1 + p.net_eq_return) / 12 - p.eq_vol ** 2 / 24
        sig_m = p.eq_vol / np.sqrt(12)
        S = p.net_monthly

        np.random.seed(42)
        log_r = np.random.normal(mu_m, sig_m, (p.mc_paths, p.N))
        returns = np.exp(log_r) - 1

        # Build portfolios via recurrence
        mf = np.zeros((p.mc_paths, p.N))
        for m in range(p.N):
            prev = mf[:, m - 1] if m > 0 else 0.0
            mf[:, m] = prev * (1 + returns[:, m]) + S

        cum_inv = S * np.arange(1, p.N + 1)
        total = p.principal + mf
        final_mf = mf[:, -1]
        final_total = total[:, -1]
        final_gains = final_mf - cum_inv[-1]

        # Tax
        cum_int_tax = p.gross_monthly * p.eff_tax * p.N
        ltcg = np.maximum(0, (final_gains - p.ltcg_exempt)) * p.ltcg_rate
        net_final = final_total - cum_int_tax - ltcg

        # Pure SIP MC for comparison (same random paths)
        sip_mf = np.zeros((p.mc_paths, p.N))
        for m in range(p.N):
            prev = sip_mf[:, m - 1] if m > 0 else 0.0
            sip_mf[:, m] = prev * (1 + returns[:, m]) + S
        sip_final = sip_mf[:, -1]
        sip_gains = sip_final - cum_inv[-1]
        sip_ltcg = np.maximum(0, (sip_gains - p.ltcg_exempt)) * p.ltcg_rate
        sip_net = sip_final - sip_ltcg

        # Path percentiles for fan chart
        pct_keys = [5, 10, 25, 50, 75, 90, 95]
        path_pcts = {k: np.percentile(total, k, axis=0) for k in pct_keys}
        sip_path_pcts = {k: np.percentile(sip_mf, k, axis=0) for k in pct_keys}

        # Max drawdown from total wealth paths
        dd = np.zeros(p.mc_paths)
        for i in range(p.mc_paths):
            peak = np.maximum.accumulate(total[i])
            dd[i] = ((total[i] - peak) / peak).min()

        return {
            'final_total': final_total,
            'net_final': net_final,
            'sip_final': sip_final,
            'sip_net': sip_net,
            'path_pcts': path_pcts,
            'sip_path_pcts': sip_path_pcts,
            'max_dd': dd,
            'returns': returns,
            'pct': {k: np.percentile(final_total, k) for k in pct_keys},
            'pct_net': {k: np.percentile(net_final, k) for k in pct_keys},
            'sip_pct': {k: np.percentile(sip_final, k) for k in pct_keys},
        }

    # ── Comparative Metrics ───────────────────────────────────

    def compute_all_metrics(self, ncd_df: pd.DataFrame, sip_df: pd.DataFrame,
                            fd_df: pd.DataFrame, mc: dict) -> dict:
        """Compute every metric for all systems — single source of truth."""
        p = self.p
        ncd_final = ncd_df.iloc[-1]
        sip_final = sip_df.iloc[-1]
        fd_final = fd_df.iloc[-1]

        # ── NCD System ──
        ncd_total = ncd_final['total_wealth']
        ncd_net = ncd_final['net_wealth']
        ncd_mf = ncd_final['mf_value']
        ncd_gains = ncd_final['mf_gains']
        ncd_cum_inv = ncd_final['cum_invested']
        ncd_cum_tax_int = ncd_final['cum_tax_interest']
        ncd_ltcg = ncd_final['ltcg_if_exit']
        ncd_total_tax = ncd_final['total_tax_if_exit']
        ncd_cagr_pre = (ncd_total / p.principal) ** (1 / p.horizon) - 1
        ncd_cagr_post = (ncd_net / p.principal) ** (1 / p.horizon) - 1
        ncd_multiple = ncd_total / p.principal
        ncd_real = ncd_total / (1 + p.inflation) ** p.horizon
        ncd_real_cagr = (ncd_real / p.principal) ** (1 / p.horizon) - 1

        # ── Pure SIP ──
        sip_total = sip_final['mf_value']
        sip_net = sip_final['net_value']
        sip_gains = sip_final['mf_gains']
        sip_cum_inv = sip_final['cum_invested']
        sip_ltcg = sip_final['ltcg_if_exit']
        sip_xirr = self._xirr_sip(sip_total, p.net_monthly, p.N)
        sip_real = sip_total / (1 + p.inflation) ** p.horizon

        # ── FD System ──
        fd_total = fd_final['total_wealth']
        fd_net = fd_final['net_wealth']
        fd_cagr_pre = (fd_total / p.principal) ** (1 / p.horizon) - 1
        fd_cagr_post = (fd_net / p.principal) ** (1 / p.horizon) - 1

        # ── Do Nothing (Savings Account) ──
        savings_total = p.principal * (1 + p.savings_rate) ** p.horizon
        savings_tax = (savings_total - p.principal) * p.eff_tax
        savings_net = savings_total - savings_tax

        # ── Advantages ──
        adv_vs_sip = ncd_net - sip_net
        adv_vs_fd = ncd_net - fd_net
        adv_vs_savings = ncd_net - savings_net

        # ── MC Risk Metrics ──
        final_vals = mc['final_total']
        net_vals = mc['net_final']
        cagrs = (final_vals / p.principal) ** (1 / p.horizon) - 1
        mean_cagr = cagrs.mean()
        vol_cagr = cagrs.std()
        rf = p.savings_rate

        sharpe = (mean_cagr - rf) / vol_cagr if vol_cagr > 0 else 0
        downside = cagrs[cagrs < rf] - rf
        downside_vol = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 0.001
        sortino = (mean_cagr - rf) / downside_vol

        gains_over_rf = np.maximum(0, cagrs - rf)
        losses_under_rf = np.maximum(0, rf - cagrs)
        omega = gains_over_rf.mean() / losses_under_rf.mean() if losses_under_rf.mean() > 0 else float('inf')

        med_dd = np.median(mc['max_dd'])
        calmar = mean_cagr / abs(med_dd) if med_dd != 0 else float('inf')

        var5 = np.percentile(net_vals, 5)
        cvar5 = net_vals[net_vals <= var5].mean() if np.any(net_vals <= var5) else var5

        prob_beat_sip = np.mean(mc['final_total'] > mc['sip_final'])
        prob_capital = np.mean(mc['net_final'] > p.principal)
        prob_double = np.mean(mc['final_total'] >= 2 * p.principal)
        prob_loss = np.mean(mc['net_final'] < p.principal)

        # ── Break-even equity return ──
        try:
            def _wealth(r):
                r_m = (1 + r) ** (1 / 12) - 1
                fv = p.net_monthly * ((1 + r_m) ** p.N - 1) / r_m if r_m != 0 else p.net_monthly * p.N
                tax_int = p.gross_monthly * p.eff_tax * p.N
                g = fv - p.net_monthly * p.N
                ltcg = max(0, (g - p.ltcg_exempt)) * p.ltcg_rate
                return p.principal + fv - tax_int - ltcg - p.principal
            breakeven = brentq(_wealth, -0.10, 0.40)
        except Exception:
            breakeven = 0.0

        return {
            # NCD System
            'ncd_total': ncd_total, 'ncd_net': ncd_net, 'ncd_mf': ncd_mf,
            'ncd_gains': ncd_gains, 'ncd_cum_inv': ncd_cum_inv,
            'ncd_cum_tax_int': ncd_cum_tax_int, 'ncd_ltcg': ncd_ltcg,
            'ncd_total_tax': ncd_total_tax,
            'ncd_cagr_pre': ncd_cagr_pre, 'ncd_cagr_post': ncd_cagr_post,
            'ncd_multiple': ncd_multiple,
            'ncd_real': ncd_real, 'ncd_real_cagr': ncd_real_cagr,
            # Pure SIP
            'sip_total': sip_total, 'sip_net': sip_net, 'sip_gains': sip_gains,
            'sip_cum_inv': sip_cum_inv, 'sip_ltcg': sip_ltcg, 'sip_xirr': sip_xirr,
            'sip_real': sip_real,
            # FD System
            'fd_total': fd_total, 'fd_net': fd_net,
            'fd_cagr_pre': fd_cagr_pre, 'fd_cagr_post': fd_cagr_post,
            # Savings
            'savings_total': savings_total, 'savings_net': savings_net,
            # Advantages
            'adv_vs_sip': adv_vs_sip, 'adv_vs_fd': adv_vs_fd,
            'adv_vs_savings': adv_vs_savings,
            # Risk
            'sharpe': sharpe, 'sortino': sortino, 'omega': omega, 'calmar': calmar,
            'var5': var5, 'cvar5': cvar5,
            'mean_cagr': mean_cagr, 'vol_cagr': vol_cagr,
            'prob_beat_sip': prob_beat_sip, 'prob_capital': prob_capital,
            'prob_double': prob_double, 'prob_loss': prob_loss,
            'med_dd': med_dd, 'breakeven': breakeven,
        }

    def _xirr_sip(self, final_val, monthly_sip, n_months) -> float:
        """Approximate XIRR for a regular SIP ending with final_val."""
        try:
            def npv(r):
                r_m = (1 + r) ** (1 / 12) - 1
                if r_m == 0:
                    return monthly_sip * n_months - final_val
                return monthly_sip * ((1 + r_m) ** n_months - 1) / r_m - final_val
            return brentq(npv, -0.20, 1.0)
        except Exception:
            return 0.0

    # ── Sensitivity ───────────────────────────────────────────

    def sensitivity_2d(self, ncd_yields: List[float], eq_returns: List[float],
                       metric: str = 'net_wealth') -> pd.DataFrame:
        """2D sensitivity matrix. metric: 'total_wealth' or 'net_wealth'."""
        p = self.p
        results = []
        for ny in ncd_yields:
            row = {}
            for er in eq_returns:
                sip_amt = p.principal * ny / 12 * (1 - p.tds_rate)
                r_m = (1 + er - p.expense_ratio) ** (1 / 12) - 1
                fv = sip_amt * ((1 + r_m) ** p.N - 1) / r_m if r_m != 0 else sip_amt * p.N
                total = p.principal + fv
                if metric == 'net_wealth':
                    tax_int = p.principal * ny * p.horizon * p.eff_tax
                    gains = fv - sip_amt * p.N
                    ltcg = max(0, (gains - p.ltcg_exempt)) * p.ltcg_rate
                    val = total - tax_int - ltcg
                else:
                    val = total
                row[f'{er:.0%}'] = val
            results.append(row)
        return pd.DataFrame(results, index=[f'{y:.1%}' for y in ncd_yields])

    # ── Annual Comparison Table ───────────────────────────────

    def annual_comparison(self, ncd_df, sip_df, fd_df) -> pd.DataFrame:
        """Year-end comparison across all systems."""
        p = self.p
        ncd_yr = ncd_df[ncd_df['month'] % 12 == 0].copy()
        sip_yr = sip_df[sip_df['month'] % 12 == 0].copy()
        fd_yr = fd_df[fd_df['month'] % 12 == 0].copy()

        rows = []
        for i in range(len(ncd_yr)):
            n = ncd_yr.iloc[i]
            s = sip_yr.iloc[i]
            f = fd_yr.iloc[i]
            yr = int(n['year'])
            inflation_factor = (1 + p.inflation) ** yr
            rows.append({
                'Year': yr,
                'NCD: Total Wealth': n['total_wealth'],
                'NCD: Net of Tax': n['net_wealth'],
                'Pure SIP: Value': s['mf_value'],
                'Pure SIP: Net': s['net_value'],
                'FD+Equity: Net': f['net_wealth'],
                'Advantage vs SIP': n['net_wealth'] - s['net_value'],
                'Advantage vs FD': n['net_wealth'] - f['net_wealth'],
                'NCD Real Wealth': n['total_wealth'] / inflation_factor,
                'NCD CAGR': (n['total_wealth'] / p.principal) ** (1 / yr) - 1,
                'NCD Post-Tax CAGR': (n['net_wealth'] / p.principal) ** (1 / yr) - 1,
            })
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# CHART FACTORY
# ═══════════════════════════════════════════════════════════════

def _layout(fig, title='', height=480):
    fig.update_layout(
        title=dict(text=title, font=dict(color=C['gold'], size=13, family='Inter')),
        paper_bgcolor=C['card'], plot_bgcolor=C['card'],
        font=dict(color=C['text2'], family='JetBrains Mono, monospace', size=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
        height=height, margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor=C['gold_border'], borderwidth=1, font=dict(size=9)),
        hoverlabel=dict(bgcolor=C['bg2'], font_size=10, font_family='JetBrains Mono'),
    )
    return fig


def chart_wealth_path(ncd_df, sip_df, fd_df, p):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ncd_df['month'], y=ncd_df['total_wealth'], name='NCD System (Total)',
                             line=dict(color=C['gold'], width=2.5),
                             hovertemplate='Month %{x}<br>%{y:,.0f}<extra>NCD System</extra>'))
    fig.add_trace(go.Scatter(x=ncd_df['month'], y=ncd_df['mf_value'], name='NCD: MF Component',
                             line=dict(color=C['cyan'], width=1.5, dash='dot'),
                             hovertemplate='Month %{x}<br>%{y:,.0f}<extra>MF Only</extra>'))
    fig.add_trace(go.Scatter(x=sip_df['month'], y=sip_df['mf_value'], name='Pure SIP',
                             line=dict(color=C['red'], width=1.8, dash='dash'),
                             hovertemplate='Month %{x}<br>%{y:,.0f}<extra>Pure SIP</extra>'))
    fig.add_trace(go.Scatter(x=fd_df['month'], y=fd_df['total_wealth'], name='FD+Equity System',
                             line=dict(color=C['orange'], width=1.5, dash='dashdot'),
                             hovertemplate='Month %{x}<br>%{y:,.0f}<extra>FD+Equity</extra>'))
    fig.add_hline(y=p.principal, line_dash='dot', line_color=C['muted'],
                  annotation_text='Capital Floor', annotation_font_color=C['muted'])
    return _layout(fig, 'Wealth Accumulation — All Systems Compared', 500)


def chart_net_comparison(ncd_df, sip_df, fd_df, p):
    ncd_yr = ncd_df[ncd_df['month'] % 12 == 0]
    sip_yr = sip_df[sip_df['month'] % 12 == 0]
    fd_yr = fd_df[fd_df['month'] % 12 == 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ncd_yr['year'], y=ncd_yr['net_wealth'], name='NCD System (Net)',
                             mode='lines+markers', line=dict(color=C['gold'], width=2.5), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=sip_yr['year'], y=sip_yr['net_value'], name='Pure SIP (Net)',
                             mode='lines+markers', line=dict(color=C['red'], width=2, dash='dash'), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=fd_yr['year'], y=fd_yr['net_wealth'], name='FD+Equity (Net)',
                             mode='lines+markers', line=dict(color=C['orange'], width=1.5, dash='dashdot'), marker=dict(size=5)))

    savings_line = [p.principal * (1 + p.savings_rate) ** yr for yr in range(1, p.horizon + 1)]
    fig.add_trace(go.Scatter(x=list(range(1, p.horizon + 1)), y=savings_line, name='Savings Account',
                             line=dict(color=C['muted'], width=1.2, dash='dot')))

    fig.add_hline(y=p.principal, line_dash='dot', line_color=C['muted'])
    fig.update_layout(yaxis_title='Net-of-Tax Value (₹)')
    return _layout(fig, 'Post-Tax Net Wealth — Comparative', 480)


def chart_advantage_bar(annual_comp):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=annual_comp['Year'], y=annual_comp['Advantage vs SIP'],
                         name='vs Pure SIP', marker_color=C['gold'], opacity=0.85))
    fig.add_trace(go.Bar(x=annual_comp['Year'], y=annual_comp['Advantage vs FD'],
                         name='vs FD+Equity', marker_color=C['cyan'], opacity=0.7))
    fig.update_layout(barmode='group', yaxis_title='Advantage (₹)')
    return _layout(fig, 'NCD System Advantage Over Alternatives (Net of Tax)', 400)


def chart_tax_waterfall(metrics, p):
    annual_int = p.principal * p.ncd_rate * p.horizon
    labels = ['NCD Interest\nIncome', 'Equity MF\nGains', 'Tax on\nInterest', 'LTCG\nTax', 'Net\nWealth']
    vals = [annual_int, metrics['ncd_gains'], -metrics['ncd_cum_tax_int'], -metrics['ncd_ltcg'], metrics['ncd_net']]
    measures = ['relative', 'relative', 'relative', 'relative', 'total']

    fig = go.Figure(go.Waterfall(
        x=labels, y=vals, measure=measures,
        connector=dict(line=dict(color=C['muted'], width=1)),
        decreasing=dict(marker=dict(color=C['red'])),
        increasing=dict(marker=dict(color=C['green'])),
        totals=dict(marker=dict(color=C['gold'])),
        textposition='outside',
        text=[fmt_smart(abs(v)) for v in vals],
        textfont=dict(color=C['text2'], size=9),
    ))
    return _layout(fig, 'Tax Waterfall — NCD System (Income → Tax → Net Wealth)', 420)


def chart_tax_comparison_bar(metrics, p):
    labels = ['NCD System', 'Pure SIP', 'FD+Equity']
    total_tax = [metrics['ncd_total_tax'], metrics['sip_ltcg'], metrics['fd_total'] - metrics['fd_net']]
    net_wealth = [metrics['ncd_net'], metrics['sip_net'], metrics['fd_net']]

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Total Tax Paid', 'Net Wealth After Tax'))
    fig.add_trace(go.Bar(x=labels, y=total_tax, marker_color=[C['gold'], C['red'], C['orange']],
                         text=[fmt_smart(v) for v in total_tax], textposition='outside',
                         textfont=dict(size=9, color=C['text2']), showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=net_wealth, marker_color=[C['gold'], C['red'], C['orange']],
                         text=[fmt_smart(v) for v in net_wealth], textposition='outside',
                         textfont=dict(size=9, color=C['text2']), showlegend=False), row=1, col=2)
    return _layout(fig, '', 400)


def chart_mc_fan(mc, p, system='ncd'):
    pcts = mc['path_pcts'] if system == 'ncd' else mc['sip_path_pcts']
    months = np.arange(1, p.N + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([months, months[::-1]]),
                             y=np.concatenate([pcts[95], pcts[5][::-1]]),
                             fill='toself', fillcolor='rgba(255,195,0,0.06)',
                             line=dict(color='rgba(0,0,0,0)'), name='5th–95th', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=np.concatenate([months, months[::-1]]),
                             y=np.concatenate([pcts[75], pcts[25][::-1]]),
                             fill='toself', fillcolor='rgba(255,195,0,0.12)',
                             line=dict(color='rgba(0,0,0,0)'), name='25th–75th', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=months, y=pcts[50], name='Median',
                             line=dict(color=C['gold'], width=2.5)))
    if system == 'ncd':
        fig.add_hline(y=p.principal, line_dash='dot', line_color=C['muted'], annotation_text='Capital Floor')
    label = 'NCD System' if system == 'ncd' else 'Pure SIP'
    return _layout(fig, f'Monte Carlo Fan Chart — {label} ({p.mc_paths:,} Paths)', 460)


def chart_mc_hist_compare(mc, p):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mc['net_final'] / 1e5, nbinsx=70, name='NCD System (Net)',
                               marker_color=C['gold'], opacity=0.65))
    fig.add_trace(go.Histogram(x=mc['sip_net'] / 1e5, nbinsx=70, name='Pure SIP (Net)',
                               marker_color=C['red'], opacity=0.45))
    fig.update_layout(barmode='overlay', xaxis_title='Terminal Net Wealth (₹ Lakhs)', yaxis_title='Frequency')
    return _layout(fig, 'Terminal Wealth Distribution — NCD System vs Pure SIP (Net of Tax)', 420)


def chart_sensitivity(sens_df, title):
    z = sens_df.values / 1e5
    fig = go.Figure(go.Heatmap(
        z=z, x=sens_df.columns.tolist(), y=sens_df.index.tolist(),
        colorscale=[[0, '#1B2A4A'], [0.5, '#FFC300'], [1, '#FF5252']],
        text=[[f'₹{v:.1f}L' for v in row] for row in z],
        texttemplate='%{text}', textfont=dict(size=10, color='white'),
        colorbar=dict(title='₹ Lakhs'),
    ))
    fig.update_layout(xaxis_title='Equity Return (p.a.)', yaxis_title='NCD Yield (p.a.)')
    return _layout(fig, title, 440)


def chart_real_wealth(ncd_df, sip_df, p):
    ncd_yr = ncd_df[ncd_df['month'] % 12 == 0].copy()
    sip_yr = sip_df[sip_df['month'] % 12 == 0].copy()
    yrs = ncd_yr['year'].values
    infl = (1 + p.inflation) ** yrs

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs, y=ncd_yr['total_wealth'].values / infl, name='NCD System (Real)',
                             mode='lines+markers', line=dict(color=C['gold'], width=2.5), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=yrs, y=sip_yr['mf_value'].values / infl, name='Pure SIP (Real)',
                             mode='lines+markers', line=dict(color=C['red'], width=2, dash='dash'), marker=dict(size=5)))
    fig.add_hline(y=p.principal, line_dash='dot', line_color=C['muted'], annotation_text='Original Capital')
    fig.update_layout(yaxis_title="Real Value (Today's ₹)")
    return _layout(fig, f'Inflation-Adjusted Wealth ({fmt_pct(p.inflation, 0)} p.a.)', 420)


# ═══════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title='Vriddhi — Hemrek Capital', page_icon='◈', layout='wide', initial_sidebar_state='expanded')
    inject_css()

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""<div style="text-align:center;padding:10px 0 14px 0;">
            <div style="font-family:'Inter';font-weight:800;font-size:1.1rem;color:{C['gold']};letter-spacing:0.04em;">◈ वृद्धि VRIDDHI</div>
            <div style="font-family:'JetBrains Mono';font-size:0.62rem;color:{C['muted']};letter-spacing:0.12em;margin-top:3px;">HEMREK CAPITAL</div>
        </div><div style="height:1px;background:{C['gold_border']};margin:0 0 14px 0;"></div>""", unsafe_allow_html=True)

        st.markdown(f"<p style='color:{C['gold']};font-size:0.72rem;font-weight:700;letter-spacing:0.1em;'>NCD PARAMETERS</p>", unsafe_allow_html=True)
        principal = st.number_input('Investment Amount (₹)', value=10_00_000, step=1_00_000, format='%d')
        ncd_rate = st.slider('NCD Coupon Rate (%)', 6.0, 16.0, 11.8, 0.1) / 100
        horizon = st.slider('Horizon (Years)', 3, 20, 10)

        st.markdown(f"<p style='color:{C['gold']};font-size:0.72rem;font-weight:700;letter-spacing:0.1em;margin-top:14px;'>EQUITY ASSUMPTIONS</p>", unsafe_allow_html=True)
        eq_return = st.slider('Expected Equity Return (%)', 5.0, 22.0, 13.0, 0.5) / 100
        eq_vol = st.slider('Equity Volatility (%)', 8.0, 35.0, 18.0, 0.5) / 100
        expense_ratio = st.slider('MF Expense Ratio (%)', 0.0, 2.5, 0.5, 0.1) / 100

        st.markdown(f"<p style='color:{C['gold']};font-size:0.72rem;font-weight:700;letter-spacing:0.1em;margin-top:14px;'>TAX (INDIA)</p>", unsafe_allow_html=True)
        tax_slab = st.selectbox('Income Tax Slab', [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], index=6,
                                format_func=lambda x: f'{x:.0%}')
        ltcg_rate = st.number_input('LTCG Rate (%)', value=12.5, step=0.5) / 100
        fd_rate = st.slider('FD Rate for Comparison (%)', 4.0, 9.0, 7.0, 0.25) / 100

        st.markdown(f"<p style='color:{C['gold']};font-size:0.72rem;font-weight:700;letter-spacing:0.1em;margin-top:14px;'>ENVIRONMENT</p>", unsafe_allow_html=True)
        inflation = st.slider('Inflation (%)', 2.0, 10.0, 6.0, 0.5) / 100
        mc_paths = st.select_slider('Monte Carlo Paths', [1000, 2500, 5000, 10000, 25000], value=5000)

    # ── BUILD ─────────────────────────────────────────────────
    p = Params(principal=principal, ncd_rate=ncd_rate, horizon=horizon,
               eq_return=eq_return, eq_vol=eq_vol, expense_ratio=expense_ratio,
               tax_slab=tax_slab, ltcg_rate=ltcg_rate, fd_rate=fd_rate,
               inflation=inflation, mc_paths=mc_paths)

    eng = Engine(p)
    ncd_df = eng.simulate_ncd_system()
    sip_df = eng.simulate_pure_sip()
    fd_df = eng.simulate_fd_reinvest()

    # ── HEADER ────────────────────────────────────────────────
    st.markdown(f"""<div class="hdr">
        <h1>◈ वृद्धि VRIDDHI</h1>
        <div class="sub">NCD + EQUITY REINVESTMENT SYSTEM — ADAM DEEP LITERATURE ENGINE</div>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # KPI STRIP — System Output at a Glance
    # ══════════════════════════════════════════════════════════
    nf = ncd_df.iloc[-1]
    sf = sip_df.iloc[-1]
    adv = nf['net_wealth'] - sf['net_value']

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric('Total Wealth', fmt_smart(nf['total_wealth']), fmt_x(nf['total_wealth'] / p.principal))
    k2.metric('Net of All Tax', fmt_smart(nf['net_wealth']),
              fmt_pct((nf['net_wealth'] / p.principal) ** (1 / p.horizon) - 1) + ' CAGR')
    k3.metric('MF Portfolio Built', fmt_smart(nf['mf_value']), f'{fmt_smart(nf["mf_gains"])} gains')
    k4.metric('Monthly SIP Generated', fmt_inr(p.net_monthly), f'from {fmt_pct(p.ncd_rate, 1)} yield')
    k5.metric('NCD Principal', fmt_smart(p.principal), 'Returned at maturity')
    k6.metric('Advantage vs SIP', fmt_smart(adv), 'net of tax')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        '◈ OVERVIEW', '📊 PROJECTIONS', '🎲 STOCHASTIC',
        '🏛️ TAX ENGINE', '⚡ SENSITIVITY', '🛡️ RISK', '📖 ADAM'
    ])

    # ──────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ──────────────────────────────────────────────────────────
    with t1:
        st.markdown('## System Thesis')
        st.markdown(f"""<div class="thesis">
            <strong>Mechanism:</strong> Deploy {fmt_smart(p.principal)} into NCDs yielding {fmt_pct(p.ncd_rate, 1)} p.a.
            → Monthly interest of {fmt_inr(p.gross_monthly)} ({fmt_inr(p.net_monthly)} post-TDS)
            → Systematically invested into equity MFs at {fmt_pct(p.eq_return, 0)} expected return
            → At maturity: NCD principal returned in full + equity portfolio worth {fmt_smart(nf['mf_value'])}
            <br><br>
            <strong>Result after {p.horizon} years:</strong> Total wealth of {fmt_smart(nf['total_wealth'])} ({fmt_x(nf['total_wealth']/p.principal)} multiple).
            After all taxes: {fmt_smart(nf['net_wealth'])} — a post-tax CAGR of {fmt_pct((nf['net_wealth']/p.principal)**(1/p.horizon)-1)} on original capital.
            The NCD system outperforms a pure SIP by {fmt_smart(adv)} because the {fmt_smart(p.principal)} principal is preserved and returned.
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns(2)
        with cl:
            st.plotly_chart(chart_wealth_path(ncd_df, sip_df, fd_df, p), use_container_width=True)
        with cr:
            st.plotly_chart(chart_net_comparison(ncd_df, sip_df, fd_df, p), use_container_width=True)

        # ── Comparative Scorecard ──
        st.markdown('## Comparative Scorecard at Year {}'.format(p.horizon))
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.markdown(f"""<div style="text-align:center;padding:12px;background:{C['card']};border:1px solid {C['gold_border']};border-radius:8px;">
            <div style="color:{C['gold']};font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;">NCD System</div>
            <div style="color:{C['gold']};font-family:'JetBrains Mono';font-size:1.3rem;font-weight:700;margin:6px 0;">{fmt_smart(nf['net_wealth'])}</div>
            <div style="color:{C['text2']};font-size:0.75rem;">CAGR: {fmt_pct((nf['net_wealth']/p.principal)**(1/p.horizon)-1)} · Tax: {fmt_smart(nf['total_tax_if_exit'])}</div>
        </div>""", unsafe_allow_html=True)
        sc2.markdown(f"""<div style="text-align:center;padding:12px;background:{C['card']};border:1px solid {C['border']};border-radius:8px;">
            <div style="color:{C['red']};font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;">Pure SIP</div>
            <div style="color:{C['red']};font-family:'JetBrains Mono';font-size:1.3rem;font-weight:700;margin:6px 0;">{fmt_smart(sf['net_value'])}</div>
            <div style="color:{C['text2']};font-size:0.75rem;">LTCG Tax: {fmt_smart(sf['ltcg_if_exit'])} · No principal</div>
        </div>""", unsafe_allow_html=True)
        ff = fd_df.iloc[-1]
        sc3.markdown(f"""<div style="text-align:center;padding:12px;background:{C['card']};border:1px solid {C['border']};border-radius:8px;">
            <div style="color:{C['orange']};font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;">FD + Equity ({fmt_pct(p.fd_rate,0)})</div>
            <div style="color:{C['orange']};font-family:'JetBrains Mono';font-size:1.3rem;font-weight:700;margin:6px 0;">{fmt_smart(ff['net_wealth'])}</div>
            <div style="color:{C['text2']};font-size:0.75rem;">Lower yield → smaller SIP corpus</div>
        </div>""", unsafe_allow_html=True)
        sav_net = p.principal * (1 + p.savings_rate) ** p.horizon
        sav_tax = (sav_net - p.principal) * p.eff_tax
        sc4.markdown(f"""<div style="text-align:center;padding:12px;background:{C['card']};border:1px solid {C['border']};border-radius:8px;">
            <div style="color:{C['muted']};font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;">Savings ({fmt_pct(p.savings_rate,0)})</div>
            <div style="color:{C['muted']};font-family:'JetBrains Mono';font-size:1.3rem;font-weight:700;margin:6px 0;">{fmt_smart(sav_net - sav_tax)}</div>
            <div style="color:{C['text2']};font-size:0.75rem;">Idle capital benchmark</div>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(chart_real_wealth(ncd_df, sip_df, p), use_container_width=True)

    # ──────────────────────────────────────────────────────────
    # TAB 2 — PROJECTIONS
    # ──────────────────────────────────────────────────────────
    with t2:
        st.markdown('## Deterministic Projections')

        # Closed-form validation
        cf_val = eng.closed_form()
        sim_val = nf['mf_value']
        delta = abs(cf_val - sim_val)
        st.markdown(f"""<div class="adam">
            <span class="tag">Adam · Closed-Form Validation</span>
            Ordinary Annuity: <code>V(N) = S·[(1+r)<sup>N</sup>-1]/r</code> where S={fmt_inr(p.net_monthly, True)}, r={p.monthly_eq_r:.8f}, N={p.N}<br>
            Closed-form: <code>{fmt_inr(cf_val, True)}</code> | Recursive: <code>{fmt_inr(sim_val, True)}</code> | Delta: <code>{fmt_inr(delta, True)}</code> {'✓' if delta < 1 else '✗'}
        </div>""", unsafe_allow_html=True)

        annual_comp = eng.annual_comparison(ncd_df, sip_df, fd_df)
        st.plotly_chart(chart_advantage_bar(annual_comp), use_container_width=True)

        # Year-end table
        st.markdown('### Year-End Comparison Table')
        disp = annual_comp.copy()
        money_cols = [c for c in disp.columns if c != 'Year' and 'CAGR' not in c]
        for col in money_cols:
            disp[col] = disp[col].apply(lambda x: fmt_smart(x))
        for col in [c for c in disp.columns if 'CAGR' in c]:
            disp[col] = disp[col].apply(lambda x: fmt_pct(x))
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Regime scenarios
        st.markdown('### Regime Scenarios')
        scenarios = {'Bear (7%)': 0.07, 'Below Avg (10%)': 0.10, 'Base (13%)': 0.13,
                     'Bull (16%)': 0.16, 'Euphoric (19%)': 0.19}
        fig_reg = go.Figure()
        colors_reg = [C['red'], C['orange'], C['gold'], C['green'], C['cyan']]
        for i, (name, ret) in enumerate(scenarios.items()):
            df_s = eng.simulate_ncd_system(eq_return=ret - p.expense_ratio)
            fig_reg.add_trace(go.Scatter(
                x=df_s['month'], y=df_s['total_wealth'], name=name,
                line=dict(color=colors_reg[i], width=2.5 if 'Base' in name else 1.5,
                          dash='solid' if 'Base' in name else 'dot')))
        fig_reg.add_hline(y=p.principal, line_dash='dot', line_color=C['muted'])
        st.plotly_chart(_layout(fig_reg, 'Total Wealth Under Different Equity Return Regimes', 460), use_container_width=True)

        scols = st.columns(5)
        for i, (name, ret) in enumerate(scenarios.items()):
            fv = eng.closed_form(eq_return=ret - p.expense_ratio)
            tw = p.principal + fv
            scols[i].metric(name, fmt_smart(tw), fmt_pct((tw / p.principal) ** (1 / p.horizon) - 1) + ' CAGR')

    # ──────────────────────────────────────────────────────────
    # TAB 3 — STOCHASTIC
    # ──────────────────────────────────────────────────────────
    with t3:
        st.markdown('## Monte Carlo Simulation')
        st.markdown(f"""<div class="adam">
            <span class="tag">Adam · GBM Framework</span>
            Equity: <code>dS = μ·S·dt + σ·S·dW</code> with drift correction <code>μ_adj = ln(1+r)/12 - σ²/24</code><br>
            Paths: <code>{p.mc_paths:,}</code> | σ_annual: <code>{fmt_pct(p.eq_vol)}</code> | σ_monthly: <code>{fmt_pct(p.eq_vol/np.sqrt(12), 4)}</code> | Seed: 42<br>
            Same random paths used for NCD system and Pure SIP to ensure paired comparison.
        </div>""", unsafe_allow_html=True)

        with st.spinner(f'Simulating {p.mc_paths:,} paths...'):
            mc = eng.monte_carlo()
            metrics = eng.compute_all_metrics(ncd_df, sip_df, fd_df, mc)

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(chart_mc_fan(mc, p, 'ncd'), use_container_width=True)
        with col_r:
            st.plotly_chart(chart_mc_fan(mc, p, 'sip'), use_container_width=True)

        st.plotly_chart(chart_mc_hist_compare(mc, p), use_container_width=True)

        st.markdown('### Distribution Comparison')
        d1, d2, d3, d4 = st.columns(4)
        d1.metric('NCD Median (Net)', fmt_smart(mc['pct_net'][50]))
        d2.metric('SIP Median (Net)', fmt_smart(np.percentile(mc['sip_net'], 50)))
        d3.metric('NCD 5th Pctl', fmt_smart(mc['pct_net'][5]))
        d4.metric('SIP 5th Pctl', fmt_smart(np.percentile(mc['sip_net'], 5)))

        d5, d6, d7, d8 = st.columns(4)
        d5.metric('NCD 95th Pctl', fmt_smart(mc['pct_net'][95]))
        d6.metric('SIP 95th Pctl', fmt_smart(np.percentile(mc['sip_net'], 95)))
        d7.metric('P(NCD Beats SIP)', fmt_pct(metrics['prob_beat_sip'], 1))
        d8.metric('P(Capital Preserved)', fmt_pct(metrics['prob_capital'], 1))

        # Percentile table
        st.markdown('### Percentile Table (Net of Tax)')
        pct_rows = []
        for k in [5, 10, 25, 50, 75, 90, 95]:
            pct_rows.append({
                'Percentile': f'{k}th',
                'NCD System': fmt_smart(mc['pct_net'][k]),
                'Pure SIP': fmt_smart(np.percentile(mc['sip_net'], k)),
                'Advantage': fmt_smart(mc['pct_net'][k] - np.percentile(mc['sip_net'], k)),
            })
        st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────
    # TAB 4 — TAX ENGINE
    # ──────────────────────────────────────────────────────────
    with t4:
        st.markdown('## Tax Analysis — India')
        st.markdown(f"""<div class="adam">
            <span class="tag">Adam · Tax Regime</span>
            <strong>NCD Interest:</strong> Income from Other Sources → Slab {fmt_pct(p.tax_slab, 0)} + Cess {fmt_pct(p.cess, 0)} = Effective {fmt_pct(p.eff_tax, 1)}. TDS at {fmt_pct(p.tds_rate, 0)}.<br>
            <strong>Equity LTCG:</strong> {fmt_pct(p.ltcg_rate, 1)} on gains > {fmt_inr(p.ltcg_exempt)}/year (Budget 2024).<br>
            <strong>Equity STCG:</strong> {fmt_pct(p.stcg_rate, 0)} flat (Budget 2024).<br>
            <strong>LTCG Harvesting:</strong> Annual redemption within exemption saves up to {fmt_inr(p.ltcg_exempt * p.ltcg_rate)} per year.
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(chart_tax_waterfall(metrics, p), use_container_width=True)
        st.plotly_chart(chart_tax_comparison_bar(metrics, p), use_container_width=True)

        # Detailed breakdown
        st.markdown('### Tax Breakdown — Comparative')
        tax_comp = {
            'Component': ['Gross Interest Income', 'Tax on Interest (Cum.)', 'TDS Deducted (Cum.)',
                          'Additional Tax (Cum.)', 'Equity MF Gains', 'LTCG on Equity',
                          'Total Tax Liability', 'Net Wealth'],
            'NCD System': [
                fmt_inr(p.principal * p.ncd_rate * p.horizon),
                fmt_inr(metrics['ncd_cum_tax_int']),
                fmt_inr(p.gross_monthly * p.tds_rate * p.N),
                fmt_inr(metrics['ncd_cum_tax_int'] - p.gross_monthly * p.tds_rate * p.N),
                fmt_inr(metrics['ncd_gains']),
                fmt_inr(metrics['ncd_ltcg']),
                fmt_inr(metrics['ncd_total_tax']),
                fmt_inr(metrics['ncd_net']),
            ],
            'Pure SIP': ['—', '—', '—', '—',
                fmt_inr(metrics['sip_gains']),
                fmt_inr(metrics['sip_ltcg']),
                fmt_inr(metrics['sip_ltcg']),
                fmt_inr(metrics['sip_net']),
            ],
            'FD+Equity': [
                fmt_inr(p.principal * p.fd_rate * p.horizon),
                fmt_inr(p.principal * p.fd_rate * p.horizon * p.eff_tax),
                fmt_inr(p.principal * p.fd_rate / 12 * p.tds_rate * p.N),
                fmt_inr(p.principal * p.fd_rate * p.horizon * p.eff_tax - p.principal * p.fd_rate / 12 * p.tds_rate * p.N),
                fmt_inr(fd_df.iloc[-1]['mf_gains']),
                fmt_inr(fd_df.iloc[-1]['ltcg_if_exit']),
                fmt_inr(metrics['fd_total'] - metrics['fd_net']),
                fmt_inr(metrics['fd_net']),
            ],
        }
        st.dataframe(pd.DataFrame(tax_comp), use_container_width=True, hide_index=True)

        st.markdown(f"""<div class="thesis">
            <strong>Tax Cost of the Yield Arbitrage:</strong> The NCD system pays {fmt_smart(metrics['ncd_total_tax'])} in total tax
            versus {fmt_smart(metrics['sip_ltcg'])} for a Pure SIP — a difference of {fmt_smart(metrics['ncd_total_tax'] - metrics['sip_ltcg'])}.
            This additional tax is the price of capital preservation. It is more than offset by the {fmt_smart(p.principal)} principal
            returned at maturity, resulting in a net advantage of {fmt_smart(metrics['adv_vs_sip'])}.
        </div>""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # TAB 5 — SENSITIVITY
    # ──────────────────────────────────────────────────────────
    with t5:
        st.markdown('## Sensitivity Analysis')

        ncd_yields = [0.08, 0.09, 0.10, 0.118, 0.13, 0.14, 0.15]
        eq_returns = [0.07, 0.10, 0.13, 0.16, 0.19]

        sens_net = eng.sensitivity_2d(ncd_yields, eq_returns, 'net_wealth')
        sens_gross = eng.sensitivity_2d(ncd_yields, eq_returns, 'total_wealth')

        cl, cr = st.columns(2)
        with cl:
            st.plotly_chart(chart_sensitivity(sens_gross, 'Total Wealth (Pre-Tax)'), use_container_width=True)
        with cr:
            st.plotly_chart(chart_sensitivity(sens_net, 'Net Wealth (Post-Tax)'), use_container_width=True)

        # Horizon sensitivity
        st.markdown('### Horizon Sensitivity')
        h_rows = []
        for h in range(3, 21):
            tp = Params(principal=p.principal, ncd_rate=p.ncd_rate, horizon=h,
                        eq_return=p.eq_return, expense_ratio=p.expense_ratio,
                        tax_slab=p.tax_slab, ltcg_rate=p.ltcg_rate)
            te = Engine(tp)
            fv = te.closed_form()
            tw = tp.principal + fv
            cum_tax_int = tp.gross_monthly * tp.eff_tax * tp.N
            gains = fv - tp.net_monthly * tp.N
            ltcg = max(0, (gains - tp.ltcg_exempt)) * tp.ltcg_rate
            net = tw - cum_tax_int - ltcg
            # Pure SIP comparison
            sip_fv = te.closed_form()
            sip_gains = sip_fv - tp.net_monthly * tp.N
            sip_ltcg = max(0, (sip_gains - tp.ltcg_exempt)) * tp.ltcg_rate
            sip_net = sip_fv - sip_ltcg
            h_rows.append({'horizon': h, 'total': tw, 'net': net,
                           'cagr': (tw / tp.principal) ** (1 / h) - 1,
                           'multiple': tw / tp.principal,
                           'sip_net': sip_net, 'advantage': net - sip_net})

        h_df = pd.DataFrame(h_rows)
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=h_df['horizon'], y=h_df['net'] / 1e5, name='NCD Net Wealth',
                               marker_color=[C['gold'] if h == p.horizon else C['card'] for h in h_df['horizon']],
                               text=[fmt_x(m) for m in h_df['multiple']], textposition='outside',
                               textfont=dict(color=C['text2'], size=8)))
        fig_h.add_trace(go.Scatter(x=h_df['horizon'], y=h_df['sip_net'] / 1e5, name='SIP Net',
                                   mode='lines+markers', line=dict(color=C['red'], dash='dash', width=1.5)))
        fig_h.update_layout(xaxis_title='Horizon (Years)', yaxis_title='₹ Lakhs')
        st.plotly_chart(_layout(fig_h, 'Net Wealth by Horizon — NCD System vs Pure SIP', 420), use_container_width=True)

        # Break-even
        st.markdown(f"""<div class="adam">
            <span class="tag">Adam · Break-Even Analysis</span>
            Solving <code>P + V(N) - Tax = P</code> via Brent's method:<br>
            Break-even equity return = <code>{fmt_pct(metrics['breakeven'], 2)}</code><br>
            Any equity return above this preserves capital after all taxes. Below it, the system still returns the NCD principal — the equity portfolio is the only component at risk.
        </div>""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # TAB 6 — RISK
    # ──────────────────────────────────────────────────────────
    with t6:
        st.markdown('## Risk Metrics & Assessment')

        st.markdown('### Quantitative Risk — NCD System (from Monte Carlo)')
        r1, r2, r3, r4 = st.columns(4)
        r1.metric('Sharpe Ratio', f'{metrics["sharpe"]:.3f}')
        r2.metric('Sortino Ratio', f'{metrics["sortino"]:.3f}')
        r3.metric('Omega Ratio', f'{metrics["omega"]:.3f}')
        r4.metric('Calmar Ratio', f'{metrics["calmar"]:.3f}')

        r5, r6, r7, r8 = st.columns(4)
        r5.metric('VaR (5%)', fmt_smart(metrics['var5']))
        r6.metric('CVaR (5%)', fmt_smart(metrics['cvar5']))
        r7.metric('Annualised Vol', fmt_pct(metrics['vol_cagr']))
        r8.metric('Median Max DD', fmt_pct(metrics['med_dd']))

        r9, r10, r11, r12 = st.columns(4)
        r9.metric('P(Capital Loss)', fmt_pct(metrics['prob_loss']))
        r10.metric('P(2x Capital)', fmt_pct(metrics['prob_double']))
        r11.metric('Break-Even Return', fmt_pct(metrics['breakeven']))
        r12.metric('Mean CAGR (MC)', fmt_pct(metrics['mean_cagr']))

        st.markdown(f"""<div class="adam">
            <span class="tag">Adam · Risk Derivations</span>
            <strong>Sharpe</strong> = (E[CAGR] - r_f) / σ_CAGR = ({fmt_pct(metrics['mean_cagr'])} - {fmt_pct(p.savings_rate, 0)}) / {fmt_pct(metrics['vol_cagr'])} = <code>{metrics['sharpe']:.4f}</code><br>
            <strong>Sortino</strong> = (E[CAGR] - r_f) / σ_downside = <code>{metrics['sortino']:.4f}</code> (penalises only returns below risk-free)<br>
            <strong>Omega</strong> = E[gains above r_f] / E[losses below r_f] = <code>{metrics['omega']:.4f}</code> (>1 favourable)<br>
            <strong>VaR (5%)</strong>: 95% confidence the net terminal wealth ≥ <code>{fmt_smart(metrics['var5'])}</code><br>
            <strong>CVaR (5%)</strong>: Expected value in the worst 5% of outcomes = <code>{fmt_smart(metrics['cvar5'])}</code>
        </div>""", unsafe_allow_html=True)

        # Drawdown histogram
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Histogram(x=mc['max_dd'] * 100, nbinsx=60, marker_color=C['red'], opacity=0.6))
        fig_dd.add_vline(x=metrics['med_dd'] * 100, line_dash='dash', line_color=C['gold'],
                         annotation_text=f'Median: {fmt_pct(metrics["med_dd"], 1)}')
        fig_dd.update_layout(xaxis_title='Max Drawdown (%)', yaxis_title='Frequency')
        st.plotly_chart(_layout(fig_dd, 'Maximum Drawdown Distribution (Total Wealth)', 380), use_container_width=True)

        # Qualitative risks
        st.markdown('### Qualitative Risk Decomposition')
        risks = [
            ('Credit / Default Risk', 'CRITICAL', C['red'],
             'NCD issuer default would result in partial or total principal loss. This is the system\'s existential risk. '
             'Mitigation: diversify across 5–10 issuers, minimum AA- rating, prefer secured NCDs, quarterly monitoring, '
             'limit single-issuer exposure to 20%.'),
            ('Equity Drawdown Risk', 'HIGH', C['orange'],
             f'MF portfolio median max drawdown: {fmt_pct(metrics["med_dd"], 1)} in simulation. Since the SIP is funded '
             'from interest (not savings), dollar-cost averaging naturally buys more units during drawdowns.'),
            ('Interest Rate Risk', 'LOW (if HTM)', C['green'],
             'Irrelevant if NCDs are held to maturity — the coupon is fixed regardless of rate environment. '
             'Risk materialises only if forced early exit is needed.'),
            ('Inflation Risk', 'MEDIUM', C['gold'],
             f'At {fmt_pct(p.inflation, 0)} inflation, real NCD coupon value declines. The equity component hedges '
             'this — historically, Indian equities have beaten inflation over 10+ year periods.'),
            ('Tax Policy Risk', 'MEDIUM', C['gold'],
             'Budget 2024 raised LTCG from 10% to 12.5%. Further changes are possible. The system\'s structure '
             '(interest taxation + LTCG) means tax policy affects both income streams.'),
            ('Reinvestment Risk', 'LOW–MEDIUM', C['cyan'],
             'At NCD maturity, equivalent yields may be unavailable. Implement a ladder strategy with staggered '
             'maturities to smooth this risk across rate cycles.'),
        ]
        for name, sev, color, desc in risks:
            st.markdown(f"""<div class="risk-card" style="border-left:3px solid {color};">
                <div class="rt" style="color:{color};">{name} — {sev}</div>
                <div class="rb">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # TAB 7 — ADAM LITERATURE
    # ──────────────────────────────────────────────────────────
    with t7:
        st.markdown('## Adam Deep Literature — Mathematical Foundations')
        st.markdown(f"""<div class="adam">
            <span class="tag">Epistemological Framework</span>
            Adam operates as Vriddhi's epistemological engine. Every computation carries its mathematical 
            provenance. Nothing is asserted without derivation. The name <em>Vriddhi</em> (वृद्धि) — "growth, 
            interest earned on capital" — is the precise Sanskrit financial term for what this system does: 
            converts debt yield into equity wealth.
        </div>""", unsafe_allow_html=True)

        st.markdown('### 1. The Yield Arbitrage Theorem')
        ncd_tax_cost = p.ncd_rate * p.horizon * p.eff_tax
        st.markdown(f"""<div class="adam">
            <span class="tag">Theorem 1</span>
            <strong>Claim:</strong> For NCD yield r_d, horizon T, and effective tax τ, the NCD system dominates 
            a pure SIP by <code>P·(1 - r_d·T·τ)</code> after tax.<br><br>
            <strong>Proof:</strong> Both systems build identical MF portfolios (same S, same r_e). The only 
            difference is the NCD principal P returned at maturity, less cumulative interest tax:<br>
            <code>Advantage = P - P·r_d·T·τ = P·(1 - r_d·T·τ)</code><br>
            = {fmt_inr(p.principal)} × (1 - {p.ncd_rate}×{p.horizon}×{p.eff_tax:.4f})<br>
            = {fmt_inr(p.principal)} × (1 - {ncd_tax_cost:.4f}) = <code>{fmt_inr(p.principal * (1 - ncd_tax_cost))}</code><br><br>
            System is advantageous when <code>r_d·T·τ < 1</code>. Current: <code>{ncd_tax_cost:.4f} < 1</code> ✓<br>
            Actual computed advantage: <code>{fmt_inr(metrics['adv_vs_sip'])}</code> (includes LTCG differential) <strong>∎</strong>
        </div>""", unsafe_allow_html=True)

        st.markdown('### 2. Ordinary Annuity Closed-Form')
        st.markdown(f"""<div class="adam">
            <span class="tag">Derivation</span>
            Recurrence: <code>V(n) = V(n-1)·(1+r) + S</code>, V(0) = 0<br>
            Expanding: V(N) = S·Σ(1+r)^k for k=0..N-1 = <code>S·[(1+r)^N - 1]/r</code><br><br>
            S = {fmt_inr(p.net_monthly, True)} | r = {p.monthly_eq_r:.10f} | N = {p.N}<br>
            V({p.N}) = <code>{fmt_inr(cf_val, True)}</code><br>
            Simulation: <code>{fmt_inr(sim_val, True)}</code> | Δ = <code>{fmt_inr(delta, True)}</code> ✓
        </div>""", unsafe_allow_html=True)

        st.markdown('### 3. GBM Drift Correction')
        mu_adj = np.log(1 + p.net_eq_return) / 12 - p.eq_vol ** 2 / 24
        sig_m = p.eq_vol / np.sqrt(12)
        st.markdown(f"""<div class="adam">
            <span class="tag">Jensen's Inequality Correction</span>
            Naive drift μ/12 produces biased expected returns under lognormal multiplicative process.
            The correction ensures <code>E[exp(r_t)] = (1+r)^(1/12)</code>:<br><br>
            <code>μ_adj = ln(1+r)/12 - σ²/24 = {mu_adj:.8f}</code><br>
            <code>σ_m = σ/√12 = {sig_m:.8f}</code><br><br>
            MC Median vs Deterministic: {fmt_smart(mc['pct'][50])} vs {fmt_smart(nf['total_wealth'])} (Δ {abs(mc['pct'][50]-nf['total_wealth'])/nf['total_wealth']:.2%})<br>
            The median falling below the mean is expected — lognormal sums are positively skewed.
        </div>""", unsafe_allow_html=True)

        st.markdown('### 4. Net Wealth Derivation')
        gross_int_total = p.principal * p.ncd_rate * p.horizon
        st.markdown(f"""<div class="adam">
            <span class="tag">Complete Tax-Adjusted Formula</span>
            <code>W_net = P + V(N) - Tax_int - Tax_LTCG</code><br><br>
            Tax_int = P·r_d·T·τ = {fmt_inr(p.principal)}×{p.ncd_rate}×{p.horizon}×{p.eff_tax:.4f} = <code>{fmt_inr(metrics['ncd_cum_tax_int'])}</code><br>
            Tax_LTCG = max(0, V(N)-S·N-Exempt)·τ_ltcg = max(0, {fmt_inr(nf['mf_value'])}-{fmt_inr(nf['cum_invested'])}-{fmt_inr(p.ltcg_exempt)})×{p.ltcg_rate} = <code>{fmt_inr(metrics['ncd_ltcg'])}</code><br><br>
            W_net = {fmt_inr(p.principal)} + {fmt_inr(nf['mf_value'])} - {fmt_inr(metrics['ncd_cum_tax_int'])} - {fmt_inr(metrics['ncd_ltcg'])} = <code>{fmt_inr(metrics['ncd_net'])}</code><br>
            Post-Tax CAGR = ({fmt_inr(metrics['ncd_net'])}/{fmt_inr(p.principal)})^(1/{p.horizon}) - 1 = <code>{fmt_pct(metrics['ncd_cagr_post'])}</code>
        </div>""", unsafe_allow_html=True)

        st.markdown('### 5. Distributional Properties')
        skew = stats.skew(mc['final_total'])
        kurt = stats.kurtosis(mc['final_total'])
        st.markdown(f"""<div class="adam">
            <span class="tag">Terminal Distribution Analysis</span>
            Under GBM, each SIP instalment grows lognormally. Their sum has no closed-form distribution → Monte Carlo required.<br><br>
            Mean: <code>{fmt_smart(mc['final_total'].mean())}</code> | Std: <code>{fmt_smart(mc['final_total'].std())}</code><br>
            Skewness: <code>{skew:.4f}</code> (positive → right tail extends further → favourable asymmetry)<br>
            Excess Kurtosis: <code>{kurt:.4f}</code> ({'leptokurtic → fat tails' if kurt > 0 else 'platykurtic → thin tails'})<br><br>
            P(Total > {fmt_smart(p.principal)}): <code>{fmt_pct(metrics['prob_capital'])}</code> | 
            P(Total > {fmt_smart(2*p.principal)}): <code>{fmt_pct(metrics['prob_double'])}</code> | 
            P(NCD > SIP): <code>{fmt_pct(metrics['prob_beat_sip'])}</code>
        </div>""", unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────
    st.markdown(f"""<div style="text-align:center;padding:28px 0 14px 0;margin-top:36px;border-top:1px solid {C['gold_border']};">
        <div style="font-family:'Inter';font-weight:800;font-size:0.85rem;color:{C['gold']};letter-spacing:0.06em;">◈ वृद्धि VRIDDHI</div>
        <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:{C['muted']};margin-top:3px;letter-spacing:0.1em;">HEMREK CAPITAL — ADAM DEEP LITERATURE ENGINE</div>
        <div style="font-family:'JetBrains Mono';font-size:0.58rem;color:{C['muted']};margin-top:6px;">वृद्धि — Growth. Interest. The yield that capital generates.</div>
    </div>""", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
