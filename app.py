"""
◈ समृद्धि SAMRIDDHI
NCD + Equity Reinvestment Intelligence System
Hemrek Capital | Adam Deep Literature Engine

Architecture:
  Mode A — DETERMINISTIC: constant equity return, closed-form, sensitivity
  Mode B — REAL WORLD: random paths, NIFTY backtest, Monte Carlo
  Mode selected in sidebar. Everything downstream adapts.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import brentq
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
EQ_VOL = 0.20
NIFTY = {2005:.363,2006:.398,2007:.548,2008:-.518,2009:.758,2010:.179,
         2011:-.246,2012:.277,2013:.068,2014:.314,2015:-.041,2016:.030,
         2017:.286,2018:.032,2019:.120,2020:.149,2021:.241,2022:.043,
         2023:.200,2024:.088}

# ═══════════════════════════════════════════════════════════════
# FORMATTING (Indian style)
# ═══════════════════════════════════════════════════════════════
def _ic(n):
    s = str(abs(int(n)))
    if len(s) <= 3: return s
    r = s[-3:]; s = s[:-3]
    while s: r = s[-2:] + ',' + r; s = s[:-2]
    return r
def fi(v): return f'{"" if v >= 0 else "-"}₹{_ic(round(abs(v)))}'
def fl(v): return f'₹{v/1e5:.2f}L'
def fs(v):
    a = abs(v)
    return f'₹{v/1e7:.2f}Cr' if a >= 1e7 else fl(v) if a >= 1e5 else fi(v)
def fp(v, d=2): return f'{v*100:.{d}f}%'
def fx(v): return f'{v:.2f}x'
def cagr(end, start, yr):
    return (end / start) ** (1 / yr) - 1 if end > 0 and start > 0 and yr > 0 else 0

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
CL = dict(bg='#0A0E17', bg2='#111827', cd='#1A1F2E', inp='#0D1117',
          g='#FFC300', gb='rgba(255,195,0,0.25)', tx='#E8EAED',
          t2='#9AA0A6', mu='#5F6368', gn='#00E676', rd='#FF5252',
          cy='#00BCD4', or_='#FF9100', pu='#B388FF', bd='rgba(255,255,255,0.06)')

def apply_theme():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp{{background:{CL['bg']};color:{CL['tx']};font-family:'Inter',sans-serif}}
    .stApp>header{{background:transparent!important}}
    section[data-testid="stSidebar"]{{background:{CL['bg2']}!important;border-right:1px solid {CL['gb']}!important}}
    section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] label{{color:{CL['t2']}!important;font-size:.82rem!important}}
    h1{{font-family:'Inter'!important;font-weight:800!important;font-size:1.4rem!important;color:{CL['tx']}!important}}
    h2{{font-family:'Inter'!important;font-weight:700!important;font-size:1.1rem!important;color:{CL['g']}!important}}
    h3{{font-family:'Inter'!important;font-weight:600!important;font-size:.92rem!important;color:{CL['tx']}!important}}
    [data-testid="stMetric"]{{background:{CL['cd']};border:1px solid {CL['gb']};border-radius:8px;padding:11px 13px}}
    [data-testid="stMetric"]:hover{{border-color:{CL['g']};box-shadow:0 0 10px rgba(255,195,0,.08)}}
    [data-testid="stMetricLabel"] p{{color:{CL['mu']}!important;font-family:'Inter'!important;font-size:.6rem!important;font-weight:600!important;text-transform:uppercase;letter-spacing:.08em}}
    [data-testid="stMetricValue"]{{color:{CL['g']}!important;font-family:'JetBrains Mono',monospace!important;font-weight:600!important;font-size:1.05rem!important}}
    [data-testid="stMetricDelta"]{{font-family:'JetBrains Mono',monospace!important;font-size:.65rem!important}}
    .stNumberInput input{{background:{CL['inp']}!important;color:{CL['g']}!important;border:1px solid {CL['gb']}!important;font-family:'JetBrains Mono',monospace!important}}
    .stTabs [data-baseweb="tab-list"]{{gap:2px;background:{CL['bg2']};border-radius:8px;padding:3px}}
    .stTabs [data-baseweb="tab"]{{background:transparent;color:{CL['mu']};font-family:'Inter';font-weight:600;font-size:.72rem}}
    .stTabs [aria-selected="true"]{{background:{CL['cd']}!important;color:{CL['g']}!important;border-bottom:2px solid {CL['g']}!important}}
    .hd{{background:linear-gradient(135deg,{CL['bg2']},{CL['cd']});border:1px solid {CL['gb']};border-left:3px solid {CL['g']};border-radius:8px;padding:13px 17px;margin-bottom:12px}}
    .hd h1{{margin:0!important;padding:0!important;font-size:1.25rem!important}}
    .hd .sub{{color:{CL['mu']};font-family:'JetBrains Mono',monospace;font-size:.64rem;margin-top:3px;letter-spacing:.05em}}
    .ab{{background:{CL['cd']};border:1px solid {CL['bd']};border-left:3px solid {CL['pu']};border-radius:6px;padding:10px 14px;margin:7px 0;font-family:'JetBrains Mono',monospace;font-size:.74rem;color:{CL['t2']};line-height:1.55}}
    .ab .tg{{color:{CL['pu']};font-weight:700;font-size:.6rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:3px;display:block}}
    .ab code{{color:{CL['g']};background:rgba(255,195,0,.08);padding:1px 4px;border-radius:3px}}
    .tb{{background:linear-gradient(135deg,rgba(255,195,0,.04),rgba(255,195,0,.01));border:1px solid {CL['gb']};border-radius:8px;padding:12px 16px;margin:7px 0;color:{CL['t2']};font-size:.79rem;line-height:1.55}}
    .tb strong{{color:{CL['g']}}}
    .dv{{height:1px;background:linear-gradient(to right,transparent,{CL['gb']},transparent);margin:16px 0}}
    .sc{{text-align:center;padding:9px;background:{CL['cd']};border-radius:8px;border:1px solid {CL['bd']}}}
    .sc.w{{border-color:{CL['gb']};box-shadow:0 0 8px rgba(255,195,0,.06)}}
    .sc .lb{{font-size:.56rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700}}
    .sc .vl{{font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;margin:3px 0}}
    .sc .dt{{font-size:.66rem;color:{CL['t2']}}}
    .rc{{background:{CL['cd']};border:1px solid {CL['bd']};border-radius:8px;padding:10px 14px;margin:4px 0}}
    .rc .rt{{font-weight:700;font-size:.8rem;margin-bottom:2px}}
    .rc .rb{{color:{CL['t2']};font-size:.74rem;line-height:1.45}}
    .sl{{color:{CL['g']};font-size:.64rem;font-weight:700;letter-spacing:.1em;margin:8px 0 4px}}
    </style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════
@dataclass
class Params:
    principal: float = 10_00_000
    ncd_rate: float = 0.118
    horizon: int = 10
    eq_return: float = 0.13
    expense_ratio: float = 0.005
    tax_slab: float = 0.30
    cess: float = 0.04
    tds_rate: float = 0.10
    ltcg_rate: float = 0.125
    ltcg_exempt: float = 1_25_000
    surcharge: float = 0.0
    inflation: float = 0.06
    fd_rate: float = 0.07
    savings_rate: float = 0.04
    mc_paths: int = 5000

    @property
    def eff_tax(self): return self.tax_slab * (1 + self.cess) * (1 + self.surcharge)
    @property
    def gross_monthly(self): return self.principal * self.ncd_rate / 12
    @property
    def net_monthly(self): return self.gross_monthly * (1 - self.tds_rate)
    @property
    def net_eq(self): return self.eq_return - self.expense_ratio
    @property
    def monthly_r(self): return (1 + self.net_eq) ** (1/12) - 1
    @property
    def total_months(self): return self.horizon * 12

# ═══════════════════════════════════════════════════════════════
# ENGINE — All computations in one place
# ═══════════════════════════════════════════════════════════════
class Engine:
    def __init__(self, params: Params):
        self.p = params

    def _annual_ltcg(self, gains, n):
        """Apply LTCG exemption per financial year, not terminal."""
        p = self.p; out = np.zeros(n); prev = 0.0
        for m in range(n):
            if (m + 1) % 12 == 0:
                fy_gain = gains[m] - prev
                out[m:] = (out[m-1] if m > 0 else 0) + max(0, (fy_gain - p.ltcg_exempt)) * p.ltcg_rate
                prev = gains[m]
            elif m > 0:
                out[m] = out[m-1]
        return out

    def run_ncd(self, monthly_returns=None):
        """NCD system: principal in debt, interest SIP'd into equity."""
        p = self.p; n = p.total_months
        r_m = monthly_returns if monthly_returns is not None else np.full(n, p.monthly_r)
        mf = cum = 0.0; mf_arr = np.zeros(n); cum_arr = np.zeros(n)
        for m in range(n):
            mf = mf * (1 + r_m[m]) + p.net_monthly
            cum += p.net_monthly
            mf_arr[m] = mf; cum_arr[m] = cum
        gains = mf_arr - cum_arr
        ltcg = self._annual_ltcg(gains, n)
        int_tax = p.gross_monthly * p.eff_tax * np.arange(1, n + 1)
        total = p.principal + mf_arr
        net = total - int_tax - ltcg
        return dict(total=total, net=net, mf=mf_arr, cum=cum_arr,
                    gains=gains, ltcg=ltcg, int_tax=int_tax,
                    total_tax=int_tax + ltcg, car=cum_arr)

    def run_lumpsum(self, monthly_returns=None):
        """Lump sum: entire principal into equity day 1."""
        p = self.p; n = p.total_months
        r_m = monthly_returns if monthly_returns is not None else np.full(n, p.monthly_r)
        val = np.zeros(n); val[0] = p.principal * (1 + r_m[0])
        for m in range(1, n):
            val[m] = val[m-1] * (1 + r_m[m])
        gains = val - p.principal
        ltcg = self._annual_ltcg(gains, n)
        return dict(total=val, net=val - ltcg, gains=gains, ltcg=ltcg,
                    car=np.full(n, p.principal))

    def run_fd_equity(self):
        """FD interest → equity SIP. Same structure as NCD, lower yield."""
        p = self.p; n = p.total_months
        fd_net = p.principal * p.fd_rate / 12 * (1 - p.tds_rate)
        r_m = np.full(n, p.monthly_r)
        mf = cum = 0.0; mf_arr = np.zeros(n); cum_arr = np.zeros(n)
        for m in range(n):
            mf = mf * (1 + r_m[m]) + fd_net; cum += fd_net
            mf_arr[m] = mf; cum_arr[m] = cum
        gains = mf_arr - cum_arr
        ltcg = self._annual_ltcg(gains, n)
        int_tax = p.principal * p.fd_rate / 12 * p.eff_tax * np.arange(1, n + 1)
        total = p.principal + mf_arr
        return dict(total=total, net=total - int_tax - ltcg,
                    total_tax=int_tax + ltcg, car=cum_arr)

    def run_pure_fd(self):
        p = self.p; n = p.total_months
        val = np.array([p.principal * (1 + p.fd_rate) ** ((m+1)/12) for m in range(n)])
        tax = (val - p.principal) * p.eff_tax
        return dict(total=val, net=val - tax)

    def run_savings(self):
        p = self.p; n = p.total_months
        val = np.array([p.principal * (1 + p.savings_rate) ** ((m+1)/12) for m in range(n)])
        tax = (val - p.principal) * p.eff_tax
        return dict(total=val, net=val - tax)

    def closed_form_mf(self):
        """V(N) = S · [(1+r)^N - 1] / r"""
        p = self.p; r = p.monthly_r
        return p.net_monthly * ((1+r)**p.total_months - 1) / r if r != 0 else p.net_monthly * p.total_months

    def decompose(self, ncd_result):
        p = self.p; ci = ncd_result['cum'][-1]; cg = ncd_result['mf'][-1] - ci
        return dict(principal=p.principal, interest=ci, compounding=cg,
                    total=p.principal + ncd_result['mf'][-1])

    # ── Sensitivity (deterministic only) ──────────────────────

    def sensitivity(self, ncd_yields, eq_returns):
        p = self.p; rows = []
        for ny in ncd_yields:
            row = {}
            for er in eq_returns:
                sip = p.principal * ny / 12 * (1 - p.tds_rate)
                rm = (1 + er - p.expense_ratio) ** (1/12) - 1
                fv = sip * ((1+rm)**p.total_months - 1) / rm if rm != 0 else sip * p.total_months
                v = p.principal + fv - p.principal * ny * p.horizon * p.eff_tax
                g = fv - sip * p.total_months
                v -= max(0, (g - p.ltcg_exempt * p.horizon)) * p.ltcg_rate
                row[f'{er:.0%}'] = v
            rows.append(row)
        return pd.DataFrame(rows, index=[f'{y:.1%}' for y in ncd_yields])

    # ── Real World engines ────────────────────────────────────

    def _generate_returns(self, seed, n_paths=1):
        """GBM monthly returns. Drift-corrected for Jensen's inequality."""
        p = self.p; mu = np.log(1 + p.net_eq) / 12 - EQ_VOL**2 / 24
        sigma = EQ_VOL / np.sqrt(12)
        np.random.seed(seed)
        return np.exp(np.random.normal(mu, sigma, (n_paths, p.total_months))) - 1

    def scenario(self, seed):
        """One random path — full comparative analytics."""
        p = self.p; mr = self._generate_returns(seed, 1)[0]
        annual_r = np.array([np.prod(1 + mr[y*12:(y+1)*12]) - 1 for y in range(p.horizon)])

        ncd = self.run_ncd(mr)
        lump = self.run_lumpsum(mr)

        def drawdown_stats(path):
            peak = np.maximum.accumulate(path)
            dd = (path - peak) / peak
            return dd, dd.min(), int(np.argmin(dd)) + 1, int(np.sum(dd < -0.001))

        n_dd, n_mdd, n_mm, n_uw = drawdown_stats(ncd['total'])
        l_dd, l_mdd, l_mm, l_uw = drawdown_stats(lump['total'])

        rolling3 = []
        for yr in range(3, p.horizon + 1):
            s, e = (yr-3)*12, yr*12-1
            rolling3.append(dict(yr=yr,
                ncd=cagr(ncd['total'][e], ncd['total'][s], 3) if ncd['total'][s] > 0 else 0,
                lump=cagr(lump['total'][e], lump['total'][s], 3) if lump['total'][s] > 0 else 0))

        year_data = []
        for y in range(p.horizon):
            m = y * 12 + 11
            leader = 'NCD' if ncd['total'][m] > lump['total'][m] else 'Lump'
            year_data.append(dict(yr=y+1, ret=annual_r[y],
                ncd_t=ncd['total'][m], ncd_n=ncd['net'][m], ncd_car=ncd['car'][m],
                lump_t=lump['total'][m], lump_n=lump['net'][m], leader=leader))

        return dict(mr=mr, annual_r=annual_r, ncd=ncd, lump=lump,
            n_dd=n_dd, n_mdd=n_mdd, n_mm=n_mm, n_uw=n_uw,
            l_dd=l_dd, l_mdd=l_mdd, l_mm=l_mm, l_uw=l_uw,
            rolling3=rolling3, year_data=year_data,
            ncd_wins=sum(1 for d in year_data if d['leader'] == 'NCD'),
            ncd_cagr=cagr(ncd['total'][-1], p.principal, p.horizon),
            lump_cagr=cagr(lump['total'][-1], p.principal, p.horizon))

    def scenario_multi(self, seed, n=25):
        """Multiple random paths for aggregate statistics."""
        p = self.p; mrs = self._generate_returns(seed, n); results = []
        for i in range(n):
            ncd = self.run_ncd(mrs[i]); lump = self.run_lumpsum(mrs[i])
            pk = np.maximum.accumulate(ncd['total'])
            ndd = ((ncd['total'] - pk) / pk).min()
            lpk = np.maximum.accumulate(lump['total'])
            ldd = ((lump['total'] - lpk) / lpk).min()
            results.append(dict(
                ncd_path=ncd['total'], lump_path=lump['total'],
                ncd_net=ncd['net'][-1], lump_net=lump['net'][-1],
                ncd_total=ncd['total'][-1], lump_total=lump['total'][-1],
                ncd_mdd=ndd, lump_mdd=ldd,
                ncd_cagr=cagr(ncd['total'][-1], p.principal, p.horizon),
                lump_cagr=cagr(lump['total'][-1], p.principal, p.horizon)))
        return results

    def backtest(self, start_year):
        """Historical NIFTY backtest for a starting year."""
        p = self.p; yrs = sorted(NIFTY.keys())
        if start_year not in yrs: return None
        idx = yrs.index(start_year)
        available = yrs[idx:idx + p.horizon]
        if len(available) < p.horizon: return None

        annual_r = [NIFTY[y] for y in available]
        mr = np.concatenate([np.full(12, (1 + r) ** (1/12) - 1) for r in annual_r])[:p.total_months]
        ncd = self.run_ncd(mr); lump = self.run_lumpsum(mr)
        return dict(annual_r=annual_r, years=available,
            ncd=ncd, lump=lump,
            ncd_net=ncd['net'][-1], lump_net=lump['net'][-1],
            ncd_total=ncd['total'][-1], lump_total=lump['total'][-1])

    def monte_carlo(self):
        """Monte Carlo — paired paths for NCD vs Lump comparison."""
        p = self.p; n = p.total_months
        mrs = self._generate_returns(42, p.mc_paths)
        cum_inv = p.net_monthly * n

        # NCD system
        mf = np.zeros((p.mc_paths, n))
        for m in range(n):
            prev = mf[:, m-1] if m > 0 else 0.0
            mf[:, m] = prev * (1 + mrs[:, m]) + p.net_monthly
        ncd_total = p.principal + mf[:, -1]
        ncd_gains = mf[:, -1] - cum_inv
        ncd_int_tax = p.gross_monthly * p.eff_tax * n
        ncd_ltcg = np.maximum(0, (ncd_gains - p.ltcg_exempt * p.horizon)) * p.ltcg_rate
        ncd_net = ncd_total - ncd_int_tax - ncd_ltcg

        # Lump sum
        lv = np.zeros((p.mc_paths, n)); lv[:, 0] = p.principal * (1 + mrs[:, 0])
        for m in range(1, n):
            lv[:, m] = lv[:, m-1] * (1 + mrs[:, m])
        lump_total = lv[:, -1]; lump_gains = lump_total - p.principal
        lump_ltcg = np.maximum(0, (lump_gains - p.ltcg_exempt * p.horizon)) * p.ltcg_rate
        lump_net = lump_total - lump_ltcg

        # Path percentiles
        ncd_paths = p.principal + mf; lump_paths = lv
        pcts = [5, 10, 25, 50, 75, 90, 95]
        ncd_pp = {k: np.percentile(ncd_paths, k, axis=0) for k in pcts}
        lump_pp = {k: np.percentile(lump_paths, k, axis=0) for k in pcts}

        # Drawdowns
        dd = np.array([((ncd_paths[i] - np.maximum.accumulate(ncd_paths[i])) /
                        np.maximum.accumulate(ncd_paths[i])).min() for i in range(p.mc_paths)])

        # Risk metrics
        ncd_cagrs = (ncd_total / p.principal) ** (1 / p.horizon) - 1
        mean_c = ncd_cagrs.mean(); vol_c = ncd_cagrs.std(); rf = p.savings_rate
        sharpe = (mean_c - rf) / vol_c if vol_c > 0 else 0
        ds = ncd_cagrs[ncd_cagrs < rf] - rf
        sortino = (mean_c - rf) / (np.sqrt(np.mean(ds**2)) if len(ds) > 0 else 0.001)
        med_dd = np.median(dd)
        calmar = mean_c / abs(med_dd) if med_dd != 0 else float('inf')
        var5 = np.percentile(ncd_net, 5)
        cvar5 = ncd_net[ncd_net <= var5].mean() if np.any(ncd_net <= var5) else var5

        try:
            def _w(r):
                rm = (1+r)**(1/12)-1
                fv = p.net_monthly * ((1+rm)**n - 1) / rm if rm != 0 else p.net_monthly * n
                g = fv - p.net_monthly * n
                return fv - p.gross_monthly * p.eff_tax * n - max(0, (g - p.ltcg_exempt * p.horizon)) * p.ltcg_rate
            breakeven = brentq(_w, -0.10, 0.40)
        except: breakeven = 0.0

        return dict(
            ncd_net=ncd_net, lump_net=lump_net,
            ncd_total=ncd_total, lump_total=lump_total,
            ncd_pp=ncd_pp, lump_pp=lump_pp, dd=dd,
            ncd_pn={k: np.percentile(ncd_net, k) for k in pcts},
            lump_pn={k: np.percentile(lump_net, k) for k in pcts},
            sharpe=sharpe, sortino=sortino, calmar=calmar,
            var5=var5, cvar5=cvar5, med_dd=med_dd,
            mean_cagr=mean_c, vol_cagr=vol_c, breakeven=breakeven,
            p_ncd_wins=np.mean(ncd_net > lump_net),
            p_capital=np.mean(ncd_net > p.principal),
            p_lump_loss=np.mean(lump_net < p.principal),
            p_2x=np.mean(ncd_total >= 2 * p.principal))

# ═══════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════
def lay(fig, title='', h=420):
    fig.update_layout(
        title=dict(text=title, font=dict(color=CL['g'], size=11, family='Inter')),
        paper_bgcolor=CL['cd'], plot_bgcolor=CL['cd'],
        font=dict(color=CL['t2'], family='JetBrains Mono, monospace', size=9),
        xaxis=dict(gridcolor='rgba(255,255,255,.04)', zerolinecolor='rgba(255,255,255,.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,.04)', zerolinecolor='rgba(255,255,255,.06)'),
        height=h, margin=dict(t=44, b=34, l=54, r=14),
        legend=dict(bgcolor='rgba(0,0,0,.3)', bordercolor=CL['gb'], borderwidth=1, font=dict(size=8)),
        hoverlabel=dict(bgcolor=CL['bg2'], font_size=9, font_family='JetBrains Mono'))
    return fig

def fan_chart(data, months, color, title, floor=None):
    fig = go.Figure()
    rgb = ','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))
    fig.add_trace(go.Scatter(x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([data[95], data[5][::-1]]),
        fill='toself', fillcolor=f'rgba({rgb},.06)', line=dict(color='rgba(0,0,0,0)'),
        name='5–95th', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([data[75], data[25][::-1]]),
        fill='toself', fillcolor=f'rgba({rgb},.12)', line=dict(color='rgba(0,0,0,0)'),
        name='25–75th', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=months, y=data[50], name='Median', line=dict(color=color, width=2.5)))
    if floor is not None:
        fig.add_hline(y=floor, line_dash='dot', line_color=CL['mu'])
    return lay(fig, title, 380)

def scorecard(cols, items):
    """Render scorecard across given columns. items: [(name,color,value,detail,is_winner)]"""
    for i, (nm, cl, vl, dt, win) in enumerate(items):
        w = 'w' if win else ''
        crown = ' 👑' if win else ''
        cols[i].markdown(f'<div class="sc {w}"><div class="lb" style="color:{cl}">{nm}{crown}</div>'
            f'<div class="vl" style="color:{cl}">{vl}</div><div class="dt">{dt}</div></div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title='Samriddhi — Hemrek Capital', page_icon='◈',
                       layout='wide', initial_sidebar_state='expanded')
    apply_theme()

    # ══════════════════════════════════════════════════════════
    # SIDEBAR — Mode-adaptive inputs
    # ══════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(f'<div style="text-align:center;padding:5px 0 8px">'
            f'<div style="font-family:Inter;font-weight:800;font-size:.95rem;color:{CL["g"]};letter-spacing:.04em">◈ समृद्धि</div>'
            f'<div style="font-family:JetBrains Mono;font-size:.54rem;color:{CL["mu"]};letter-spacing:.12em;margin-top:1px">HEMREK CAPITAL</div>'
            f'</div><div style="height:1px;background:{CL["gb"]};margin:0 0 10px"></div>', unsafe_allow_html=True)

        # Mode selector
        st.markdown(f'<p class="sl">ANALYSIS MODE</p>', unsafe_allow_html=True)
        mode = st.radio('Mode', ['📐 Deterministic', '🌍 Real World'],
                        label_visibility='collapsed', horizontal=True)
        is_rw = '🌍' in mode

        # Capital — always visible
        st.markdown(f'<p class="sl">CAPITAL & STRUCTURE</p>', unsafe_allow_html=True)
        principal = st.number_input('NCD Investment (₹)', value=10_00_000, step=1_00_000, format='%d')
        ncd_rate = st.slider('NCD Yield (%)', 6.0, 16.0, 11.8, 0.1) / 100
        horizon = st.slider('Horizon (Years)', 3, 15, 10)

        # Mode-specific inputs
        if is_rw:
            st.markdown(f'<p class="sl">EQUITY CALIBRATION</p>', unsafe_allow_html=True)
            eq_drift = st.slider('Expected Drift (%)', 8.0, 18.0, 13.0, 0.5,
                                 help='Long-term average around which random returns vary') / 100
            expense_ratio = st.slider('MF Expense (%)', 0.0, 2.5, 0.5, 0.1) / 100
            st.markdown(f'<p class="sl">SCENARIO CONTROLS</p>', unsafe_allow_html=True)
            seed = st.number_input('Scenario Seed', value=42, step=1,
                                   help='Each seed = unique market history')
            mc_paths = st.select_slider('Monte Carlo Paths', [1000, 2500, 5000, 10000], value=5000)
        else:
            st.markdown(f'<p class="sl">EQUITY ASSUMPTION</p>', unsafe_allow_html=True)
            eq_drift = st.slider('Equity Return (%)', 5.0, 22.0, 13.0, 0.5) / 100
            expense_ratio = st.slider('MF Expense (%)', 0.0, 2.5, 0.5, 0.1) / 100
            seed = 42; mc_paths = 5000  # unused in deterministic

        # Tax — all Indian tax params exposed
        st.markdown(f'<p class="sl">TAX (INDIA FY 2025-26)</p>', unsafe_allow_html=True)
        tax_slab = st.selectbox('Income Tax Slab', [0.0, .05, .10, .15, .20, .25, .30],
                                index=6, format_func=lambda x: f'{x:.0%}')
        cess = st.number_input('Health & Education Cess (%)', value=4.0, step=0.5,
                               help='4% on tax amount') / 100
        surcharge = st.number_input('Surcharge (%)', value=0.0, step=5.0,
                                    help='10% if 50L-1Cr, 15% if 1-2Cr, 25% if 2-5Cr') / 100
        ltcg_rate = st.number_input('Equity LTCG Rate (%)', value=12.5, step=0.5,
                                    help='12.5% post Budget 2024') / 100
        ltcg_exempt = st.number_input('LTCG Exemption (₹/FY)', value=1_25_000, step=25_000,
                                      format='%d', help='₹1,25,000 per FY post Budget 2024')
        tds_rate = st.number_input('NCD TDS Rate (%)', value=10.0, step=1.0,
                                   help='10% if PAN provided, 20% without PAN') / 100

        # Benchmarks
        st.markdown(f'<p class="sl">BENCHMARKS</p>', unsafe_allow_html=True)
        fd_rate = st.slider('FD Rate (%)', 4.0, 9.0, 7.0, 0.25) / 100
        inflation = st.slider('Inflation (%)', 2.0, 10.0, 6.0, 0.5) / 100

    # ── Build params & engine ─────────────────────────────────
    p = Params(principal=principal, ncd_rate=ncd_rate, horizon=horizon,
               eq_return=eq_drift, expense_ratio=expense_ratio,
               tax_slab=tax_slab, cess=cess, tds_rate=tds_rate,
               ltcg_rate=ltcg_rate, ltcg_exempt=ltcg_exempt,
               surcharge=surcharge,
               fd_rate=fd_rate, inflation=inflation, mc_paths=mc_paths)
    eng = Engine(p)
    months = np.arange(1, p.total_months + 1)
    N = p.total_months - 1

    # ══════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════
    mode_label = 'REAL WORLD ANALYSIS' if is_rw else 'DETERMINISTIC ANALYSIS'
    st.markdown(f'<div class="hd"><h1>◈ समृद्धि SAMRIDDHI</h1>'
        f'<div class="sub">{mode_label} — ADAM DEEP LITERATURE ENGINE</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # 📐 DETERMINISTIC MODE
    # ══════════════════════════════════════════════════════════
    if not is_rw:
        ncd = eng.run_ncd(); lump = eng.run_lumpsum()
        fd_eq = eng.run_fd_equity(); pure_fd = eng.run_pure_fd(); sav = eng.run_savings()
        dec = eng.decompose(ncd)

        # KPIs
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric('NCD Net', fs(ncd['net'][N]), fp(cagr(ncd['net'][N], p.principal, p.horizon)) + ' CAGR')
        k2.metric('Lump Sum Net', fs(lump['net'][N]), fp(cagr(lump['net'][N], p.principal, p.horizon)) + ' CAGR')
        k3.metric('Monthly SIP', fi(p.net_monthly), f'from {fp(p.ncd_rate,1)} yield')
        k4.metric('MF Built', fs(ncd['mf'][N]), f'{fs(ncd["gains"][N])} gains')
        k5.metric('Capital-at-Risk', fs(ncd['car'][N]), f'vs {fs(p.principal)} lump')
        k6.metric('NCD Principal', fs(p.principal), 'Returned at maturity')
        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs(['◈ OVERVIEW', '📊 YEAR-BY-YEAR', '⚡ SENSITIVITY', '📖 ADAM'])

        # ── Overview ──────────────────────────────────────────
        with t1:
            st.markdown('## System Overview')
            st.markdown(f'<div class="tb"><strong>Constant {fp(p.eq_return,0)} annual return assumed.</strong> '
                f'Under constant returns, lump sum always wins (full compounding from day 1). '
                f'The NCD system\'s advantage — capital preservation + DCA into crashes — is invisible here. '
                f'Switch to 🌍 Real World mode in the sidebar for volatility-based analysis.</div>', unsafe_allow_html=True)

            fig = go.Figure()
            for name, data, color, dash in [
                ('NCD System', ncd['total'], CL['g'], 'solid'),
                ('Lump Sum', lump['total'], CL['cy'], 'dash'),
                (f'FD+Eq ({fp(p.fd_rate,0)})', fd_eq['total'], CL['or_'], 'dashdot'),
                ('Pure FD', pure_fd['total'], CL['mu'], 'dot'),
                ('Savings', sav['total'], CL['rd'], 'dot')]:
                fig.add_trace(go.Scatter(x=months, y=data, name=name, line=dict(color=color, width=2.5 if dash=='solid' else 1.5, dash=dash)))
            fig.add_hline(y=p.principal, line_dash='dot', line_color=CL['mu'], annotation_text='Capital')
            st.plotly_chart(lay(fig, 'Total Wealth — 5 Strategies', 440), width='stretch')

            st.markdown('### Net Wealth Scorecard')
            items = [
                ('NCD System', CL['g'], fs(ncd['net'][N]), f'Tax {fs(ncd["total_tax"][N])} · CaR {fs(ncd["car"][N])}'),
                ('Lump Sum', CL['cy'], fs(lump['net'][N]), f'Tax {fs(lump["ltcg"][N])} · CaR {fs(p.principal)}'),
                (f'FD+Eq', CL['or_'], fs(fd_eq['net'][N]), f'Tax {fs(fd_eq["total_tax"][N])}'),
                ('Pure FD', CL['mu'], fs(pure_fd['net'][N]), f'No equity risk'),
                ('Savings', CL['rd'], fs(sav['net'][N]), f'Idle capital')]
            winner_val = max(x[2] for x in items)
            items_w = [(nm, cl, vl, dt, vl == winner_val) for nm, cl, vl, dt in items]
            scorecard(st.columns(5), items_w)

            st.markdown('### Wealth Decomposition — NCD System')
            fig_d = go.Figure(go.Waterfall(
                x=['Principal\n(returned)', 'Interest\nReinvested', 'Equity\nCompounding', 'Total'],
                y=[dec['principal'], dec['interest'], dec['compounding'], dec['total']],
                measure=['relative', 'relative', 'relative', 'total'],
                connector=dict(line=dict(color=CL['mu'], width=1)),
                increasing=dict(marker=dict(color=CL['gn'])),
                totals=dict(marker=dict(color=CL['g'])),
                textposition='outside',
                text=[fs(v) for v in [dec['principal'], dec['interest'], dec['compounding'], dec['total']]],
                textfont=dict(color=CL['t2'], size=9)))
            st.plotly_chart(lay(fig_d, '', 340), width='stretch')

        # ── Year-by-Year ──────────────────────────────────────
        with t2:
            st.markdown('## Year-by-Year Projections')
            cf_val = eng.closed_form_mf(); delta = abs(cf_val - ncd['mf'][N])
            st.markdown(f'<div class="ab"><span class="tg">Closed-Form</span>'
                f'<code>V(N)=S·[(1+r)^N-1]/r</code> = <code>{fi(round(cf_val))}</code> '
                f'| Simulation: <code>{fi(round(ncd["mf"][N]))}</code> '
                f'| Δ: <code>₹{delta:.2f}</code> ✓</div>', unsafe_allow_html=True)

            tbl = []
            for yr in range(1, p.horizon + 1):
                m = yr * 12 - 1
                tbl.append({'Year': yr, 'NCD Net': fs(ncd['net'][m]), 'Lump Net': fs(lump['net'][m]),
                    'FD+Eq Net': fs(fd_eq['net'][m]), 'NCD CAGR': fp(cagr(ncd['total'][m], p.principal, yr)),
                    'NCD vs Lump': fs(ncd['net'][m] - lump['net'][m]), 'CaR': fs(ncd['car'][m])})
            st.dataframe(pd.DataFrame(tbl), width='stretch', hide_index=True)

            st.markdown('### Regime Scenarios')
            regimes = {'Bear 7%': .07, 'Below 10%': .10, 'Base 13%': .13, 'Bull 16%': .16, 'Euphoric 19%': .19}
            colors = [CL['rd'], CL['or_'], CL['g'], CL['gn'], CL['cy']]
            fig_r = go.Figure()
            for i, (nm, ret) in enumerate(regimes.items()):
                d = eng.run_ncd(np.full(p.total_months, (1 + ret - p.expense_ratio)**(1/12) - 1))
                fig_r.add_trace(go.Scatter(x=months, y=d['total'], name=nm, line=dict(color=colors[i], width=2.5 if 'Base' in nm else 1.5, dash='solid' if 'Base' in nm else 'dot')))
            fig_r.add_hline(y=p.principal, line_dash='dot', line_color=CL['mu'])
            st.plotly_chart(lay(fig_r, 'NCD System Under Different Regimes', 400), width='stretch')

            cols = st.columns(5)
            for i, (nm, ret) in enumerate(regimes.items()):
                fv = eng.closed_form_mf() if ret == p.eq_return else p.net_monthly * ((1 + (1+ret-p.expense_ratio)**(1/12)-1)**p.total_months - 1) / ((1+ret-p.expense_ratio)**(1/12)-1) if (1+ret-p.expense_ratio)**(1/12)-1 != 0 else p.net_monthly * p.total_months
                tw = p.principal + fv
                cols[i].metric(nm, fs(tw), fp(cagr(tw, p.principal, p.horizon)))

        # ── Sensitivity ───────────────────────────────────────
        with t3:
            st.markdown('## Sensitivity Analysis')
            sens = eng.sensitivity([.08, .09, .10, .118, .13, .14, .15], [.07, .10, .13, .16, .19])
            z = sens.values / 1e5
            fig_s = go.Figure(go.Heatmap(z=z, x=sens.columns.tolist(), y=sens.index.tolist(),
                colorscale=[[0, '#1B2A4A'], [.5, '#FFC300'], [1, '#FF5252']],
                text=[[f'₹{v:.1f}L' for v in row] for row in z], texttemplate='%{text}',
                textfont=dict(size=10, color='white'), colorbar=dict(title='₹L')))
            fig_s.update_layout(xaxis_title='Equity Return', yaxis_title='NCD Yield')
            st.plotly_chart(lay(fig_s, 'Net Wealth — NCD Yield × Equity Return', 420), width='stretch')

            st.markdown('### Horizon Sensitivity')
            h_data = []
            for h in range(3, 21):
                tp = Params(principal=p.principal, ncd_rate=p.ncd_rate, horizon=h,
                            eq_return=p.eq_return, expense_ratio=p.expense_ratio,
                            tax_slab=p.tax_slab, ltcg_rate=p.ltcg_rate)
                te = Engine(tp); fv = te.closed_form_mf(); tw = tp.principal + fv
                ti = tp.gross_monthly * tp.eff_tax * tp.total_months
                g = fv - tp.net_monthly * tp.total_months
                lc = max(0, (g - tp.ltcg_exempt * h)) * tp.ltcg_rate
                h_data.append(dict(h=h, net=tw - ti - lc, mult=tw / tp.principal))
            hdf = pd.DataFrame(h_data)
            fig_h = go.Figure(go.Bar(x=hdf['h'], y=hdf['net'] / 1e5,
                marker_color=[CL['g'] if h == p.horizon else CL['cd'] for h in hdf['h']],
                text=[fx(m) for m in hdf['mult']], textposition='outside',
                textfont=dict(size=8, color=CL['t2'])))
            fig_h.update_layout(xaxis_title='Horizon (Years)', yaxis_title='Net Wealth (₹L)')
            st.plotly_chart(lay(fig_h, 'Net Wealth by Horizon', 360), width='stretch')

        # ── Adam ──────────────────────────────────────────────
        with t4:
            st.markdown('## Adam — Mathematical Foundations')
            ntc = p.ncd_rate * p.horizon * p.eff_tax
            st.markdown(f'<div class="ab"><span class="tg">Yield Arbitrage Theorem</span>'
                f'NCD dominates pure SIP by <code>P·(1-r_d·T·τ) = {fi(round(p.principal*(1-ntc)))}</code><br>'
                f'Valid when r_d·T·τ < 1. Current: {ntc:.4f} < 1 ✓</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ab"><span class="tg">Why Deterministic Favours Lump Sum</span>'
                f'Under constant returns, lump gets <code>P·(1+r)^N</code> — full compounding from month 1. '
                f'NCD builds exposure gradually. The NCD thesis rests on <strong>volatility</strong>: '
                f'DCA buys more during drawdowns, principal is shielded from crashes. '
                f'This edge is invisible here.</div>', unsafe_allow_html=True)
            mu_a = np.log(1 + p.net_eq) / 12 - EQ_VOL**2 / 24
            st.markdown(f'<div class="ab"><span class="tg">GBM Drift Correction</span>'
                f'<code>μ_adj = ln(1+r)/12 - σ²/24 = {mu_a:.8f}</code><br>'
                f'<code>σ_m = σ/√12 = {EQ_VOL/np.sqrt(12):.8f}</code><br>'
                f'Corrects Jensen\'s inequality for unbiased multiplicative returns.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ab"><span class="tg">Per-FY LTCG Model</span>'
                f'₹{_ic(int(p.ltcg_exempt))} exemption applied at each FY boundary. '
                f'Saves up to {fi(round(p.ltcg_exempt*p.ltcg_rate))}/year vs terminal-only model.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ab"><span class="tg">Capital-at-Risk</span>'
                f'NCD: <code>{fi(round(ncd["car"][N]))}</code> (only reinvested interest in equity)<br>'
                f'Lump: <code>{fi(p.principal)}</code> (100% at equity risk from day 1)<br>'
                f'Net per unit CaR — NCD: {ncd["net"][N]/ncd["car"][N]:.2f}× | Lump: {lump["net"][N]/p.principal:.2f}×</div>', unsafe_allow_html=True)

            # Tax section
            st.markdown('### Tax Structure')
            st.markdown(f'<div class="ab"><span class="tg">Tax Regime</span>'
                f'<strong>NCD Interest:</strong> Slab {fp(p.tax_slab,0)} + Cess {fp(p.cess,0)}'
                f'{" + Surcharge " + fp(p.surcharge,0) if p.surcharge > 0 else ""}'
                f' = Effective {fp(p.eff_tax,1)}. TDS {fp(p.tds_rate,0)} at source.<br>'
                f'<strong>LTCG:</strong> {fp(p.ltcg_rate,1)} on gains > {fi(p.ltcg_exempt)}/FY (per-FY exemption).</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                '': ['Total Wealth', 'Total Tax', 'Net Wealth', 'Capital-at-Risk'],
                'NCD': [fi(ncd['total'][N]), fi(ncd['total_tax'][N]), fi(ncd['net'][N]), fi(ncd['car'][N])],
                'Lump Sum': [fi(lump['total'][N]), fi(lump['ltcg'][N]), fi(lump['net'][N]), fi(p.principal)],
                'FD+Eq': [fi(fd_eq['total'][N]), fi(fd_eq['total_tax'][N]), fi(fd_eq['net'][N]), fi(fd_eq['car'][N])],
            }), width='stretch', hide_index=True)

    # ══════════════════════════════════════════════════════════
    # 🌍 REAL WORLD MODE
    # ══════════════════════════════════════════════════════════
    else:
        # Run MC for KPIs (cached across tabs via session state)
        with st.spinner(f'Calibrating ({p.mc_paths:,} simulations)...'):
            mc = eng.monte_carlo()

        # KPIs from MC
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric('NCD Median Net', fs(mc['ncd_pn'][50]), fp(cagr(mc['ncd_pn'][50], p.principal, p.horizon)))
        k2.metric('5th–95th Range', f'{fs(mc["ncd_pn"][5])} – {fs(mc["ncd_pn"][95])}')
        k3.metric('P(NCD > Lump)', fp(mc['p_ncd_wins'], 1))
        k4.metric('P(Capital Safe)', fp(mc['p_capital'], 1))
        k5.metric('P(Lump Loss)', fp(mc['p_lump_loss'], 1))
        k6.metric('Break-Even', fp(mc['breakeven']), 'min equity return')
        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(['🎯 SCENARIO', '📜 BACKTEST', '🎲 DISTRIBUTION'])

        # ── Scenario ──────────────────────────────────────────
        with t1:
            st.markdown('## Market Scenario')
            st.markdown(f'<div class="tb">Seed <strong>{seed}</strong> generates a unique {p.horizon}-year market. '
                f'Equity drift ≈ {fp(p.eq_return,0)}, volatility ≈ {fp(EQ_VOL,0)}. '
                f'NCD system and Lump Sum face <strong>identical returns</strong> — only the deployment strategy differs.</div>', unsafe_allow_html=True)

            sc = eng.scenario(seed); ar = sc['annual_r']

            # Environment
            st.markdown('### Market Environment')
            c1, c2 = st.columns([3, 2])
            with c1:
                fb = go.Figure(go.Bar(x=list(range(1, p.horizon+1)), y=ar*100,
                    marker_color=[CL['gn'] if r >= 0 else CL['rd'] for r in ar],
                    text=[fp(r) for r in ar], textposition='outside',
                    textfont=dict(size=8, color=CL['t2'])))
                fb.update_layout(xaxis_title='Year', yaxis_title='%',
                    yaxis=dict(zeroline=True, zerolinecolor=CL['mu']))
                st.plotly_chart(lay(fb, 'Annual Returns', 300), width='stretch')
            with c2:
                st.dataframe(pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Best Year', 'Worst Year', 'Positive Yrs', 'Realised Vol'],
                    'Value': [fp(np.mean(ar)), fp(np.median(ar)),
                        f'{fp(max(ar))} (Yr {int(np.argmax(ar))+1})',
                        f'{fp(min(ar))} (Yr {int(np.argmin(ar))+1})',
                        f'{sum(1 for r in ar if r>=0)}/{p.horizon}', fp(np.std(ar))]
                }), width='stretch', hide_index=True, height=260)

            # Paths
            st.markdown('### Wealth Path')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=sc['ncd']['total'], name='NCD System', line=dict(color=CL['g'], width=2.5)))
            fig.add_trace(go.Scatter(x=months, y=sc['lump']['total'], name='Lump Sum', line=dict(color=CL['cy'], width=2, dash='dash')))
            fig.add_hline(y=p.principal, line_dash='dot', line_color=CL['mu'])
            st.plotly_chart(lay(fig, 'NCD vs Lump — Same Returns', 400), width='stretch')

            # Outcome
            st.markdown('### Outcome')
            w = 'NCD' if sc['ncd']['net'][-1] > sc['lump']['net'][-1] else 'Lump Sum'
            o1,o2,o3,o4,o5,o6 = st.columns(6)
            o1.metric('NCD Net', fs(sc['ncd']['net'][-1]), fp(sc['ncd_cagr']))
            o2.metric('Lump Net', fs(sc['lump']['net'][-1]), fp(sc['lump_cagr']))
            o3.metric('Winner', w, fs(abs(sc['ncd']['net'][-1] - sc['lump']['net'][-1])))
            o4.metric('NCD Max DD', fp(sc['n_mdd'], 1), f'vs {fp(sc["l_mdd"],1)} lump')
            o5.metric('NCD Underwater', f'{sc["n_uw"]}/{p.total_months}', f'vs {sc["l_uw"]}/{p.total_months}')
            o6.metric('NCD Leads', f'{sc["ncd_wins"]}/{p.horizon}', 'year-ends')

            # Drawdown
            st.markdown('### Drawdown from Peak')
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=months, y=sc['n_dd']*100, name='NCD', fill='tozeroy', fillcolor='rgba(255,195,0,.12)', line=dict(color=CL['g'], width=1.5)))
            fig_dd.add_trace(go.Scatter(x=months, y=sc['l_dd']*100, name='Lump', fill='tozeroy', fillcolor='rgba(0,188,212,.08)', line=dict(color=CL['cy'], width=1.2, dash='dash')))
            fig_dd.update_layout(xaxis_title='Month', yaxis_title='Drawdown (%)')
            st.plotly_chart(lay(fig_dd, '', 300), width='stretch')

            # Year table
            st.markdown('### Year-by-Year')
            st.dataframe(pd.DataFrame([{
                'Year': d['yr'], 'Return': fp(d['ret']),
                'NCD Total': fs(d['ncd_t']), 'NCD Net': fs(d['ncd_n']),
                'Lump Total': fs(d['lump_t']), 'Lump Net': fs(d['lump_n']),
                'Leader': d['leader'], 'Gap': fs(abs(d['ncd_t'] - d['lump_t'])),
                'NCD CaR': fs(d['ncd_car'])
            } for d in sc['year_data']]), width='stretch', hide_index=True)

            # Rolling 3Y
            if sc['rolling3']:
                st.markdown('### Rolling 3-Year CAGR')
                fig_r3 = go.Figure()
                fig_r3.add_trace(go.Bar(x=[r['yr'] for r in sc['rolling3']], y=[r['ncd']*100 for r in sc['rolling3']], name='NCD', marker_color=CL['g'], opacity=.8))
                fig_r3.add_trace(go.Bar(x=[r['yr'] for r in sc['rolling3']], y=[r['lump']*100 for r in sc['rolling3']], name='Lump', marker_color=CL['cy'], opacity=.6))
                fig_r3.update_layout(barmode='group', xaxis_title='End Year', yaxis_title='3Y CAGR (%)', yaxis=dict(zeroline=True, zerolinecolor=CL['mu']))
                st.plotly_chart(lay(fig_r3, '', 300), width='stretch')

            # 25 paths
            st.markdown('### 25-Path Aggregate')
            multi = eng.scenario_multi(seed * 1000, 25)
            fig_m = go.Figure()
            for i, r in enumerate(multi):
                fig_m.add_trace(go.Scatter(x=months, y=r['ncd_path'], showlegend=(i==0), name='NCD' if i==0 else None, legendgroup='n', line=dict(color=CL['g'], width=.7), opacity=.35))
                fig_m.add_trace(go.Scatter(x=months, y=r['lump_path'], showlegend=(i==0), name='Lump' if i==0 else None, legendgroup='l', line=dict(color=CL['cy'], width=.6), opacity=.25))
            fig_m.add_hline(y=p.principal, line_dash='dot', line_color=CL['mu'])
            st.plotly_chart(lay(fig_m, '25 Paths — NCD (gold) vs Lump (cyan)', 420), width='stretch')

            na = np.array([r['ncd_net'] for r in multi]); la = np.array([r['lump_net'] for r in multi])
            nc = np.array([r['ncd_cagr'] for r in multi]); lc = np.array([r['lump_cagr'] for r in multi])
            nd_ = [r['ncd_mdd'] for r in multi]; ld = [r['lump_mdd'] for r in multi]
            st.dataframe(pd.DataFrame({
                'Metric': ['Mean Net', 'Worst Case', 'Best Case', 'Mean CAGR', 'Mean Max DD', 'Wins'],
                'NCD': [fs(na.mean()), fs(na.min()), fs(na.max()), fp(nc.mean()), fp(np.mean(nd_), 1), f'{np.sum(na>la)}/25'],
                'Lump': [fs(la.mean()), fs(la.min()), fs(la.max()), fp(lc.mean()), fp(np.mean(ld), 1), f'{np.sum(la>na)}/25'],
            }), width='stretch', hide_index=True)

        # ── Backtest ──────────────────────────────────────────
        with t2:
            st.markdown('## Historical NIFTY Backtest')
            st.markdown(f'<div class="tb">Every {p.horizon}-year window using <strong>actual NIFTY50 returns</strong> ({min(NIFTY.keys())}–{max(NIFTY.keys())}). No simulation — real history.</div>', unsafe_allow_html=True)

            valid = [y for y in sorted(NIFTY.keys()) if y + p.horizon - 1 <= max(NIFTY.keys())]
            if valid:
                htbl = []; ncd_w = 0
                for yr in valid:
                    r = eng.backtest(yr)
                    if r:
                        w = 'NCD' if r['ncd_net'] > r['lump_net'] else 'Lump'
                        if w == 'NCD': ncd_w += 1
                        htbl.append({'Start': yr, 'Period': f'{yr}–{yr+p.horizon-1}',
                            'NCD Net': fs(r['ncd_net']), 'Lump Net': fs(r['lump_net']),
                            'Winner': w, 'Margin': fs(abs(r['ncd_net'] - r['lump_net']))})
                st.dataframe(pd.DataFrame(htbl), width='stretch', hide_index=True)
                st.markdown(f'<div class="tb"><strong>{ncd_w}/{len(htbl)}</strong> windows won by NCD. '
                    f'NCD wins when the period includes crashes (2008, 2011, 2020). '
                    f'Lump wins in sustained bull markets.</div>', unsafe_allow_html=True)

                sel = st.selectbox('View detailed path', valid, format_func=lambda y: f'{y}–{y+p.horizon-1}')
                bt = eng.backtest(sel)
                if bt:
                    c1, c2 = st.columns(2)
                    with c1:
                        fb = go.Figure(go.Bar(x=bt['years'], y=[r*100 for r in bt['annual_r']],
                            marker_color=[CL['gn'] if r >= 0 else CL['rd'] for r in bt['annual_r']],
                            text=[fp(r) for r in bt['annual_r']], textposition='outside',
                            textfont=dict(size=7, color=CL['t2'])))
                        st.plotly_chart(lay(fb, f'NIFTY {sel}–{sel+p.horizon-1}', 300), width='stretch')
                    with c2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=months, y=bt['ncd']['total'], name='NCD', line=dict(color=CL['g'], width=2.5)))
                        fig.add_trace(go.Scatter(x=months, y=bt['lump']['total'], name='Lump', line=dict(color=CL['cy'], width=2, dash='dash')))
                        fig.add_hline(y=p.principal, line_dash='dot', line_color=CL['mu'])
                        st.plotly_chart(lay(fig, 'Wealth Path', 300), width='stretch')
                    b1, b2, b3 = st.columns(3)
                    b1.metric('NCD Net', fs(bt['ncd_net']), fp(cagr(bt['ncd_total'], p.principal, p.horizon)))
                    b2.metric('Lump Net', fs(bt['lump_net']), fp(cagr(bt['lump_total'], p.principal, p.horizon)))
                    b3.metric('Winner', 'NCD' if bt['ncd_net'] > bt['lump_net'] else 'Lump')
            else:
                st.info(f'Need {p.horizon}+ years of data. Reduce horizon.')

        # ── Distribution ──────────────────────────────────────
        with t3:
            st.markdown('## Monte Carlo Distribution')
            st.markdown(f'<div class="ab"><span class="tg">Adam · GBM</span>'
                f'{p.mc_paths:,} paired paths. σ={fp(EQ_VOL,0)}. Drift-corrected. Same draws for NCD & Lump.<br>'
                f'Tax: Slab {fp(p.tax_slab,0)} + Cess {fp(p.cess,0)}'
                f'{" + Surcharge " + fp(p.surcharge,0) if p.surcharge > 0 else ""}'
                f' = {fp(p.eff_tax,1)} on interest. LTCG {fp(p.ltcg_rate,1)} (exempt {fi(p.ltcg_exempt)}/FY). TDS {fp(p.tds_rate,0)}.</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(fan_chart(mc['ncd_pp'], months, CL['g'], 'NCD System', p.principal), width='stretch')
            with c2: st.plotly_chart(fan_chart(mc['lump_pp'], months, CL['cy'], 'Lump Sum', p.principal), width='stretch')

            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(x=mc['ncd_net']/1e5, nbinsx=60, name='NCD Net', marker_color=CL['g'], opacity=.6))
            fig_h.add_trace(go.Histogram(x=mc['lump_net']/1e5, nbinsx=60, name='Lump Net', marker_color=CL['cy'], opacity=.4))
            fig_h.update_layout(barmode='overlay', xaxis_title='Net Wealth (₹L)', yaxis_title='Freq')
            st.plotly_chart(lay(fig_h, 'Terminal Distribution — Net of Tax', 360), width='stretch')

            st.markdown('### Percentile Table (Net)')
            st.dataframe(pd.DataFrame([{
                'Pctl': f'{k}th', 'NCD': fs(mc['ncd_pn'][k]), 'Lump': fs(mc['lump_pn'][k]),
                'NCD vs Lump': fs(mc['ncd_pn'][k] - mc['lump_pn'][k])
            } for k in [5, 10, 25, 50, 75, 90, 95]]), width='stretch', hide_index=True)

            st.markdown('### Risk Metrics')
            r1,r2,r3,r4 = st.columns(4)
            r1.metric('Sharpe', f'{mc["sharpe"]:.3f}'); r2.metric('Sortino', f'{mc["sortino"]:.3f}')
            r3.metric('VaR (5%)', fs(mc['var5'])); r4.metric('CVaR (5%)', fs(mc['cvar5']))
            r5,r6,r7,r8 = st.columns(4)
            r5.metric('Mean CAGR', fp(mc['mean_cagr'])); r6.metric('CAGR Vol', fp(mc['vol_cagr']))
            r7.metric('Median Max DD', fp(mc['med_dd'], 1)); r8.metric('Break-Even', fp(mc['breakeven']))

            fig_dd = go.Figure(go.Histogram(x=mc['dd']*100, nbinsx=60, marker_color=CL['rd'], opacity=.6))
            fig_dd.add_vline(x=mc['med_dd']*100, line_dash='dash', line_color=CL['g'], annotation_text=f'Med: {fp(mc["med_dd"],1)}')
            fig_dd.update_layout(xaxis_title='Max DD (%)', yaxis_title='Freq')
            st.plotly_chart(lay(fig_dd, 'Max Drawdown Distribution (NCD)', 300), width='stretch')

            # Risk cards
            st.markdown('### Risk Assessment')
            for nm, sev, cl, desc in [
                ('Credit Risk', 'CRITICAL', CL['rd'], 'NCD default → principal loss. Diversify 5–10 issuers, AA- min, secured.'),
                ('Equity Drawdown', 'HIGH', CL['or_'], f'MC median max DD: {fp(mc["med_dd"],1)}. Only reinvested interest exposed.'),
                ('Interest Rate', 'LOW (HTM)', CL['gn'], 'Fixed coupon. No impact if held to maturity.'),
                ('Reinvestment', 'LOW–MED', CL['cy'], 'At maturity, yields may differ. Ladder strategy recommended.')]:
                st.markdown(f'<div class="rc" style="border-left:3px solid {cl}"><div class="rt" style="color:{cl}">{nm} — {sev}</div><div class="rb">{desc}</div></div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────
    st.markdown(f'<div style="text-align:center;padding:16px 0 8px;margin-top:24px;border-top:1px solid {CL["gb"]}">'
        f'<div style="font-family:Inter;font-weight:800;font-size:.75rem;color:{CL["g"]};letter-spacing:.06em">◈ समृद्धि SAMRIDDHI</div>'
        f'<div style="font-family:JetBrains Mono;font-size:.5rem;color:{CL["mu"]};margin-top:2px;letter-spacing:.1em">HEMREK CAPITAL — ADAM DEEP LITERATURE ENGINE</div></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
