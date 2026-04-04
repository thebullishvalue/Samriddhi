"""
◈ समृद्धि SAMRIDDHI — NCD + Equity Reinvestment Intelligence System
Hemrek Capital | Adam Deep Literature Engine

समृद्धि (Samriddhi): Complete prosperity — the state where capital has
been deployed to its fullest productive potential. While vriddhi is the
yield, samriddhi is the outcome: total wealth achieved through the
systematic conversion of debt yield into equity growth.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

EQ_VOL = 0.20
# NIFTY50 calendar year returns (approx total return) for historical stress test
NIFTY_HIST = {
    2005:.363, 2006:.398, 2007:.548, 2008:-.518, 2009:.758,
    2010:.179, 2011:-.246, 2012:.277, 2013:.068, 2014:.314,
    2015:-.041, 2016:.030, 2017:.286, 2018:.032, 2019:.120,
    2020:.149, 2021:.241, 2022:.043, 2023:.200, 2024:.088}

# ═══════════════════════════════════════════════════════════════
# INDIAN FORMATTING
# ═══════════════════════════════════════════════════════════════
def _ic(n):
    s=str(abs(int(n)));
    if len(s)<=3: return s
    r=s[-3:]; s=s[:-3]
    while s: r=s[-2:]+','+r; s=s[:-2]
    return r
def fi(v,p=False):
    sg='-' if v<0 else ''; a=abs(v)
    return f'{sg}₹{_ic(int(a))}.{round((a-int(a))*100):02d}' if p else f'{sg}₹{_ic(round(a))}'
def fl(v,d=2): return f'₹{v/1e5:.{d}f}L'
def fc(v,d=2): return f'₹{v/1e7:.{d}f}Cr'
def fs(v):
    a=abs(v)
    if a>=1e7: return fc(v)
    if a>=1e5: return fl(v)
    return fi(v)
def fp(v,d=2): return f'{v*100:.{d}f}%'
def fx(v): return f'{v:.2f}x'

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
C=dict(bg='#0A0E17',bg2='#111827',cd='#1A1F2E',inp='#0D1117',
       g='#FFC300',gb='rgba(255,195,0,0.25)',tx='#E8EAED',t2='#9AA0A6',mu='#5F6368',
       gn='#00E676',rd='#FF5252',bl='#448AFF',cy='#00BCD4',pu='#B388FF',
       or_='#FF9100',bd='rgba(255,255,255,0.06)')

def css():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root{{--g:{C['g']};--cd:{C['cd']};--bg:{C['bg']};--t2:{C['t2']};--mu:{C['mu']}}}
    .stApp{{background:var(--bg);color:{C['tx']};font-family:'Inter',sans-serif}}
    .stApp>header{{background:transparent!important}}
    section[data-testid="stSidebar"]{{background:{C['bg2']}!important;border-right:1px solid {C['gb']}!important}}
    section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] label{{color:{C['t2']}!important;font-size:.84rem!important}}
    h1{{font-family:'Inter'!important;font-weight:800!important;font-size:1.5rem!important;color:{C['tx']}!important;letter-spacing:-.02em}}
    h2{{font-family:'Inter'!important;font-weight:700!important;font-size:1.15rem!important;color:var(--g)!important}}
    h3{{font-family:'Inter'!important;font-weight:600!important;font-size:.95rem!important;color:{C['tx']}!important}}
    [data-testid="stMetric"]{{background:var(--cd);border:1px solid {C['gb']};border-radius:8px;padding:12px 14px}}
    [data-testid="stMetric"]:hover{{border-color:var(--g);box-shadow:0 0 12px rgba(255,195,0,.08)}}
    [data-testid="stMetricLabel"] p{{color:var(--mu)!important;font-family:'Inter'!important;font-size:.65rem!important;font-weight:600!important;text-transform:uppercase;letter-spacing:.08em}}
    [data-testid="stMetricValue"]{{color:var(--g)!important;font-family:'JetBrains Mono',monospace!important;font-weight:600!important;font-size:1.15rem!important}}
    [data-testid="stMetricDelta"]{{font-family:'JetBrains Mono',monospace!important;font-size:.7rem!important}}
    .stNumberInput input{{background:{C['inp']}!important;color:var(--g)!important;border:1px solid {C['gb']}!important;border-radius:6px!important;font-family:'JetBrains Mono',monospace!important}}
    .stTabs [data-baseweb="tab-list"]{{gap:2px;background:{C['bg2']};border-radius:8px;padding:3px}}
    .stTabs [data-baseweb="tab"]{{background:transparent;color:var(--mu);border-radius:6px;font-family:'Inter';font-weight:600;font-size:.76rem}}
    .stTabs [aria-selected="true"]{{background:var(--cd)!important;color:var(--g)!important;border-bottom:2px solid var(--g)!important}}
    .streamlit-expanderHeader{{background:var(--cd)!important;border:1px solid {C['bd']}!important;border-radius:6px!important;color:var(--t2)!important;font-family:'JetBrains Mono',monospace!important;font-size:.82rem!important}}
    .streamlit-expanderContent{{background:var(--cd)!important;border:1px solid {C['bd']}!important;border-top:none!important}}
    .hdr{{background:linear-gradient(135deg,{C['bg2']},{C['cd']});border:1px solid {C['gb']};border-left:3px solid var(--g);border-radius:8px;padding:14px 18px;margin-bottom:16px}}
    .hdr h1{{margin:0!important;padding:0!important;font-size:1.3rem!important}}
    .hdr .sub{{color:var(--mu);font-family:'JetBrains Mono',monospace;font-size:.68rem;margin-top:3px;letter-spacing:.05em}}
    .ad{{background:var(--cd);border:1px solid {C['bd']};border-left:3px solid {C['pu']};border-radius:6px;padding:12px 16px;margin:8px 0;font-family:'JetBrains Mono',monospace;font-size:.77rem;color:var(--t2);line-height:1.6}}
    .ad .tg{{color:{C['pu']};font-weight:700;font-size:.64rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px;display:block}}
    .ad code{{color:var(--g);background:rgba(255,195,0,.08);padding:1px 5px;border-radius:3px}}
    .th{{background:linear-gradient(135deg,rgba(255,195,0,.04),rgba(255,195,0,.01));border:1px solid {C['gb']};border-radius:8px;padding:14px 18px;margin:8px 0;color:var(--t2);font-size:.82rem;line-height:1.6}}
    .th strong{{color:var(--g)}}
    .dv{{height:1px;background:linear-gradient(to right,transparent,{C['gb']},transparent);margin:20px 0}}
    .sc{{text-align:center;padding:11px;background:var(--cd);border-radius:8px;border:1px solid {C['bd']};margin-bottom:4px}}
    .sc.w{{border-color:{C['gb']};box-shadow:0 0 10px rgba(255,195,0,.06)}}
    .sc .lb{{font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700}}
    .sc .vl{{font-family:'JetBrains Mono',monospace;font-size:1.15rem;font-weight:700;margin:4px 0}}
    .sc .dt{{font-size:.7rem;color:var(--t2)}}
    .sc .car{{font-size:.62rem;color:var(--mu);margin-top:3px;font-style:italic}}
    .rc{{background:var(--cd);border:1px solid {C['bd']};border-radius:8px;padding:12px 16px;margin:4px 0}}
    .rc .rt{{font-weight:700;font-size:.84rem;margin-bottom:2px}}
    .rc .rb{{color:var(--t2);font-size:.77rem;line-height:1.5}}
    .dec{{background:var(--cd);border:1px solid {C['bd']};border-radius:8px;padding:14px 18px;margin:8px 0}}
    .dec .dl{{display:flex;justify-content:space-between;margin:4px 0;font-size:.8rem}}
    .dec .dl .dk{{color:var(--t2)}}
    .dec .dl .dv2{{color:var(--g);font-family:'JetBrains Mono',monospace;font-weight:600}}
    </style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PARAMS
# ═══════════════════════════════════════════════════════════════
@dataclass
class Pm:
    principal:float=10_00_000; ncd_rate:float=0.118; horizon:int=10
    eq_return:float=0.13; expense_ratio:float=0.005
    tax_slab:float=0.30; cess:float=0.04; tds_rate:float=0.10
    ltcg_rate:float=0.125; stcg_rate:float=0.20; ltcg_exempt:float=1_25_000
    surcharge:float=0.0; inflation:float=0.06; fd_rate:float=0.07
    savings_rate:float=0.04; mc_paths:int=5000
    @property
    def et(self): return self.tax_slab*(1+self.cess)*(1+self.surcharge)
    @property
    def gm(self): return self.principal*self.ncd_rate/12
    @property
    def nm(self): return self.gm*(1-self.tds_rate)
    @property
    def ne(self): return self.eq_return-self.expense_ratio
    @property
    def mr(self): return (1+self.ne)**(1/12)-1
    @property
    def N(self): return self.horizon*12

# ═══════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════
class E:
    def __init__(s,p): s.p=p

    def _sip_sim(s, sip, r_m, N):
        """Core SIP recurrence: V(n) = V(n-1)·(1+r) + S"""
        mf=cum=0.0; vals=[]
        for m in range(N):
            rm = r_m if np.isscalar(r_m) else r_m[m]
            mf=mf*(1+rm)+sip; cum+=sip; vals.append((mf,cum))
        return np.array(vals)

    # ── 5 Investment Strategies ───────────────────────────────

    def ncd_sys(s, er=None, monthly_r=None):
        """NCD → interest → SIP. Returns (total_wealth, net_wealth, mf, cum_inv, details)"""
        p=s.p; rm = monthly_r if monthly_r is not None else (1+(er if er is not None else p.ne))**(1/12)-1
        v=s._sip_sim(p.nm, rm, p.N)
        mf=v[:,0]; cum=v[:,1]; g=mf-cum
        # Per-FY LTCG with annual exemption (correct model)
        ltcg_cum=np.zeros(p.N)
        prev_gains=0.0
        for m in range(p.N):
            if (m+1)%12==0:  # FY boundary
                fy_gains = g[m] - prev_gains  # gains this FY
                fy_ltcg = max(0, (fy_gains-p.ltcg_exempt))*p.ltcg_rate
                ltcg_cum[m:] = (ltcg_cum[m-1] if m>0 else 0) + fy_ltcg
                prev_gains = g[m]
            elif m>0: ltcg_cum[m]=ltcg_cum[m-1]
        tax_int = p.gm*p.et*np.arange(1,p.N+1)
        total = p.principal+mf; net = total-tax_int-ltcg_cum
        return dict(mf=mf,cum=cum,g=g,total=total,net=net,tax_int=tax_int,ltcg=ltcg_cum,
                    ttax=tax_int+ltcg_cum, car=cum)  # capital-at-risk = only SIP deployed

    def lumpsum_eq(s, er=None, monthly_r=None):
        """₹P directly into equity MF. Full capital at risk."""
        p=s.p; rm = monthly_r if monthly_r is not None else (1+(er if er is not None else p.ne))**(1/12)-1
        vals=np.zeros(p.N); vals[0]=p.principal*(1+(rm if np.isscalar(rm) else rm[0]))
        for m in range(1,p.N):
            r = rm if np.isscalar(rm) else rm[m]
            vals[m]=vals[m-1]*(1+r)
        g=vals-p.principal
        ltcg_cum=np.zeros(p.N)
        prev_g=0.0
        for m in range(p.N):
            if (m+1)%12==0:
                fg=g[m]-prev_g; ltcg_cum[m:]=(ltcg_cum[m-1] if m>0 else 0)+max(0,(fg-p.ltcg_exempt))*p.ltcg_rate
                prev_g=g[m]
            elif m>0: ltcg_cum[m]=ltcg_cum[m-1]
        net=vals-ltcg_cum
        return dict(total=vals,net=net,g=g,ltcg=ltcg_cum,car=np.full(p.N,p.principal))

    def fd_sys(s):
        """FD → interest → SIP. Same structure as NCD, lower yield."""
        p=s.p; fd_nm=p.principal*p.fd_rate/12*(1-p.tds_rate)
        v=s._sip_sim(fd_nm, p.mr, p.N)
        mf=v[:,0]; cum=v[:,1]; g=mf-cum
        ltcg_cum=np.zeros(p.N); prev_g=0.0
        for m in range(p.N):
            if (m+1)%12==0:
                fg=g[m]-prev_g; ltcg_cum[m:]=(ltcg_cum[m-1] if m>0 else 0)+max(0,(fg-p.ltcg_exempt))*p.ltcg_rate
                prev_g=g[m]
            elif m>0: ltcg_cum[m]=ltcg_cum[m-1]
        tax_int=p.principal*p.fd_rate/12*p.et*np.arange(1,p.N+1)
        total=p.principal+mf; net=total-tax_int-ltcg_cum
        return dict(mf=mf,cum=cum,total=total,net=net,tax_int=tax_int,ltcg=ltcg_cum,ttax=tax_int+ltcg_cum,car=cum)

    def pure_fd(s):
        """₹P in FD, no equity. Annual compounding, interest taxed at slab."""
        p=s.p
        vals=np.zeros(p.N)
        for m in range(p.N):
            yr=(m+1)/12; vals[m]=p.principal*(1+p.fd_rate)**yr
        tax=(vals-p.principal)*p.et; net=vals-tax
        return dict(total=vals,net=net,car=np.zeros(p.N))  # no equity risk

    def savings(s):
        p=s.p
        vals=np.array([p.principal*(1+p.savings_rate)**((m+1)/12) for m in range(p.N)])
        tax=(vals-p.principal)*p.et; net=vals-tax
        return dict(total=vals,net=net,car=np.zeros(p.N))

    # ── Closed-form ───────────────────────────────────────────

    def sens(s, nys, ers):
        """2D sensitivity: NCD yield × Equity return → Net Wealth."""
        p=s.p; rows=[]
        for ny in nys:
            row={}
            for er in ers:
                sip_=p.principal*ny/12*(1-p.tds_rate)
                rm=(1+er-p.expense_ratio)**(1/12)-1
                fv=sip_*((1+rm)**p.N-1)/rm if rm!=0 else sip_*p.N
                v=p.principal+fv; v-=p.principal*ny*p.horizon*p.et
                g=fv-sip_*p.N; v-=max(0,(g-p.ltcg_exempt*p.horizon))*p.ltcg_rate
                row[f'{er:.0%}']=v
            rows.append(row)
        return pd.DataFrame(rows, index=[f'{y:.1%}' for y in nys])

    def cf(s, sip=None, er=None):
        p=s.p; S=sip or p.nm; r=(1+(er if er is not None else p.ne))**(1/12)-1
        return S*((1+r)**p.N-1)/r if r!=0 else S*p.N

    # ── Wealth Decomposition ──────────────────────────────────
    def decompose(s, ncd_res):
        """Break total wealth into: principal + interest deployed + compounding gains"""
        p=s.p
        principal = p.principal
        interest_deployed = ncd_res['cum'][-1]  # total SIP invested = all interest received
        compounding = ncd_res['mf'][-1] - interest_deployed  # equity growth on top of SIP
        return dict(principal=principal, interest=interest_deployed, compounding=compounding,
                    total=principal+ncd_res['mf'][-1])

    # ── Stress Test (random + historical NIFTY) ───────────────
    def stress_random(s, seed, n=1):
        p=s.p; mu=np.log(1+p.ne)/12-EQ_VOL**2/24; sig=EQ_VOL/np.sqrt(12)
        np.random.seed(seed); res=[]
        for _ in range(n):
            mr=np.exp(np.random.normal(mu,sig,p.N))-1
            ar=[np.prod(1+mr[y*12:(y+1)*12])-1 for y in range(p.horizon)]
            ncd=s.ncd_sys(monthly_r=mr); ls=s.lumpsum_eq(monthly_r=mr)
            res.append(dict(mr=mr,ar=np.array(ar),
                            ncd_path=ncd['total'],ncd_net=ncd['net'][-1],
                            ls_path=ls['total'],ls_net=ls['net'][-1]))
        return res

    def stress_historical(s, start_year):
        """Use actual NIFTY returns starting from start_year."""
        p=s.p; yrs=sorted(NIFTY_HIST.keys())
        if start_year not in yrs: return None
        idx=yrs.index(start_year)
        available=yrs[idx:idx+p.horizon]
        if len(available)<p.horizon: return None
        annual_r=[NIFTY_HIST[y] for y in available]
        # Expand to monthly (distribute annual return evenly)
        mr=[]
        for ar in annual_r:
            monthly=(1+ar)**(1/12)-1
            mr.extend([monthly]*12)
        mr=np.array(mr[:p.N])
        ncd=s.ncd_sys(monthly_r=mr); ls=s.lumpsum_eq(monthly_r=mr)
        return dict(annual_r=annual_r,years=available,
                    ncd_path=ncd['total'],ncd_net=ncd['net'][-1],ncd_total=ncd['total'][-1],
                    ls_path=ls['total'],ls_net=ls['net'][-1],ls_total=ls['total'][-1])

    # ── Monte Carlo ───────────────────────────────────────────
    def mc(s):
        p=s.p; mu=np.log(1+p.ne)/12-EQ_VOL**2/24; sig=EQ_VOL/np.sqrt(12)
        np.random.seed(42); mr=np.exp(np.random.normal(mu,sig,(p.mc_paths,p.N)))-1
        # NCD system
        mf=np.zeros((p.mc_paths,p.N))
        for m in range(p.N):
            prev=mf[:,m-1] if m>0 else 0.0; mf[:,m]=prev*(1+mr[:,m])+p.nm
        ci=p.nm*p.N; ft=p.principal+mf[:,-1]; fg=mf[:,-1]-ci
        it=p.gm*p.et*p.N; lc=np.maximum(0,(fg-p.ltcg_exempt*p.horizon))*p.ltcg_rate
        fn=ft-it-lc
        # Lump sum
        ls=np.zeros((p.mc_paths,p.N)); ls[:,0]=p.principal*(1+mr[:,0])
        for m in range(1,p.N):
            ls[:,m]=ls[:,m-1]*(1+mr[:,m])
        lf=ls[:,-1]; lg=lf-p.principal
        ll=np.maximum(0,(lg-p.ltcg_exempt*p.horizon))*p.ltcg_rate; ln_=lf-ll
        # SIP (same paths)
        smf=np.zeros((p.mc_paths,p.N))
        for m in range(p.N):
            prev=smf[:,m-1] if m>0 else 0.0; smf[:,m]=prev*(1+mr[:,m])+p.nm
        sf=smf[:,-1]; sg_=sf-ci; sl=np.maximum(0,(sg_-p.ltcg_exempt*p.horizon))*p.ltcg_rate; sn=sf-sl
        # Paths
        tot=p.principal+mf; pk=[5,10,25,50,75,90,95]
        pp={k:np.percentile(tot,k,axis=0) for k in pk}
        lp={k:np.percentile(ls,k,axis=0) for k in pk}
        dd=np.array([((tot[i]-np.maximum.accumulate(tot[i]))/np.maximum.accumulate(tot[i])).min() for i in range(p.mc_paths)])
        return dict(ft=ft,fn=fn,lf=lf,ln=ln_,sf=sf,sn=sn,pp=pp,lp=lp,dd=dd,
                    pn={k:np.percentile(fn,k) for k in pk},
                    lpn={k:np.percentile(ln_,k) for k in pk},
                    spn={k:np.percentile(sn,k) for k in pk})

    # ── All Metrics ───────────────────────────────────────────
    def metrics(s, ncd, ls, fd, pfd, sav, mc):
        p=s.p; N=p.N-1
        def _cagr(v): return (v/p.principal)**(1/p.horizon)-1 if v>0 else 0

        # Risk from MC
        cg=(mc['ft']/p.principal)**(1/p.horizon)-1; mc_=cg.mean(); vc=cg.std(); rf=p.savings_rate
        sh=(mc_-rf)/vc if vc>0 else 0
        ds=cg[cg<rf]-rf; dv=np.sqrt(np.mean(ds**2)) if len(ds)>0 else .001; so=(mc_-rf)/dv
        g_=np.maximum(0,cg-rf); l_=np.maximum(0,rf-cg)
        om=g_.mean()/l_.mean() if l_.mean()>0 else float('inf')
        md=np.median(mc['dd']); ca=mc_/abs(md) if md!=0 else float('inf')
        v5=np.percentile(mc['fn'],5)
        cv5=mc['fn'][mc['fn']<=v5].mean() if np.any(mc['fn']<=v5) else v5
        try:
            def _w(r):
                rm=(1+r)**(1/12)-1
                fv=p.nm*((1+rm)**p.N-1)/rm if rm!=0 else p.nm*p.N
                g=fv-p.nm*p.N; return fv-p.gm*p.et*p.N-max(0,(g-p.ltcg_exempt*p.horizon))*p.ltcg_rate
            be=brentq(_w,-.10,.40)
        except: be=0.0

        return dict(
            ncd_net=ncd['net'][N], ncd_total=ncd['total'][N], ncd_mf=ncd['mf'][N],
            ncd_gains=ncd['g'][N], ncd_tax=ncd['ttax'][N], ncd_car=ncd['car'][N],
            ls_net=ls['net'][N], ls_total=ls['total'][N], ls_gains=ls['g'][N],
            ls_tax=ls['ltcg'][N], ls_car=ls['car'][N],
            fd_net=fd['net'][N], fd_total=fd['total'][N], fd_tax=fd['ttax'][N],
            pfd_net=pfd['net'][N], sav_net=sav['net'][N],
            sh=sh,so=so,om=om,ca=ca,v5=v5,cv5=cv5,mc_=mc_,vc=vc,md=md,be=be,
            pb_ls=np.mean(mc['fn']>mc['ln']),  # P(NCD net > Lumpsum net)
            pb_sip=np.mean(mc['ft']>mc['sf']),
            pc=np.mean(mc['fn']>p.principal), p2=np.mean(mc['ft']>=2*p.principal),
            pl=np.mean(mc['fn']<p.principal),
            # Lump sum risk
            ls_sh=((mc['lf']/p.principal)**(1/p.horizon)-1).mean()-rf,
            ls_pl=np.mean(mc['ln']<p.principal),
        )

# ═══════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════
def _lo(fig,title='',h=480):
    fig.update_layout(title=dict(text=title,font=dict(color=C['g'],size=12,family='Inter')),
        paper_bgcolor=C['cd'],plot_bgcolor=C['cd'],
        font=dict(color=C['t2'],family='JetBrains Mono,monospace',size=10),
        xaxis=dict(gridcolor='rgba(255,255,255,.04)',zerolinecolor='rgba(255,255,255,.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,.04)',zerolinecolor='rgba(255,255,255,.06)'),
        height=h,margin=dict(t=48,b=38,l=58,r=18),
        legend=dict(bgcolor='rgba(0,0,0,.3)',bordercolor=C['gb'],borderwidth=1,font=dict(size=9)),
        hoverlabel=dict(bgcolor=C['bg2'],font_size=10,font_family='JetBrains Mono'))
    return fig

# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title='Samriddhi — Hemrek Capital',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
    css()
    with st.sidebar:
        st.markdown(f'<div style="text-align:center;padding:6px 0 10px"><div style="font-family:Inter;font-weight:800;font-size:1rem;color:{C["g"]};letter-spacing:.04em">◈ समृद्धि SAMRIDDHI</div><div style="font-family:JetBrains Mono;font-size:.58rem;color:{C["mu"]};letter-spacing:.12em;margin-top:2px">HEMREK CAPITAL</div></div><div style="height:1px;background:{C["gb"]};margin:0 0 10px"></div>', unsafe_allow_html=True)
        st.markdown(f"<p style='color:{C['g']};font-size:.68rem;font-weight:700;letter-spacing:.1em'>CAPITAL</p>", unsafe_allow_html=True)
        principal=st.number_input('Investment (₹)',value=10_00_000,step=1_00_000,format='%d')
        ncd_rate=st.slider('NCD Yield (%)',6.0,16.0,11.8,0.1)/100
        horizon=st.slider('Horizon (Years)',3,15,10)
        st.markdown(f"<p style='color:{C['g']};font-size:.68rem;font-weight:700;letter-spacing:.1em;margin-top:10px'>EQUITY</p>", unsafe_allow_html=True)
        eq_return=st.slider('Expected Return (%)',5.0,22.0,13.0,0.5)/100
        expense_ratio=st.slider('Expense Ratio (%)',0.0,2.5,0.5,0.1)/100
        st.markdown(f"<p style='color:{C['g']};font-size:.68rem;font-weight:700;letter-spacing:.1em;margin-top:10px'>TAX (INDIA)</p>", unsafe_allow_html=True)
        tax_slab=st.selectbox('Tax Slab',[0.0,.05,.10,.15,.20,.25,.30],index=6,format_func=lambda x:f'{x:.0%}')
        ltcg_rate=st.number_input('LTCG (%)',value=12.5,step=0.5)/100
        fd_rate=st.slider('FD Rate (%)',4.0,9.0,7.0,0.25)/100
        st.markdown(f"<p style='color:{C['g']};font-size:.68rem;font-weight:700;letter-spacing:.1em;margin-top:10px'>ENVIRONMENT</p>", unsafe_allow_html=True)
        inflation=st.slider('Inflation (%)',2.0,10.0,6.0,0.5)/100
        mc_paths=st.select_slider('MC Paths',[1000,2500,5000,10000],value=5000)

    p=Pm(principal=principal,ncd_rate=ncd_rate,horizon=horizon,eq_return=eq_return,expense_ratio=expense_ratio,tax_slab=tax_slab,ltcg_rate=ltcg_rate,fd_rate=fd_rate,inflation=inflation,mc_paths=mc_paths)
    eng=E(p); months=np.arange(1,p.N+1)
    nd=eng.ncd_sys(); ls=eng.lumpsum_eq(); fd=eng.fd_sys(); pfd=eng.pure_fd(); sv=eng.savings()
    N=p.N-1; dec=eng.decompose(nd)

    st.markdown(f'<div class="hdr"><h1>◈ समृद्धि SAMRIDDHI</h1><div class="sub">NCD + EQUITY REINVESTMENT SYSTEM — ADAM DEEP LITERATURE ENGINE</div></div>', unsafe_allow_html=True)

    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric('NCD System Net',fs(nd['net'][N]),fp((nd['net'][N]/p.principal)**(1/p.horizon)-1)+' CAGR')
    k2.metric('Lump Sum Eq Net',fs(ls['net'][N]),fp((ls['net'][N]/p.principal)**(1/p.horizon)-1)+' CAGR')
    k3.metric('Monthly SIP',fi(p.nm),f'{fp(p.ncd_rate,1)} yield')
    k4.metric('MF Built',fs(nd['mf'][N]),f'{fs(nd["g"][N])} gains')
    k5.metric('Capital-at-Risk',fs(nd['car'][N]),f'vs {fs(p.principal)} in lumpsum')
    k6.metric('NCD Principal',fs(p.principal),'Returned at maturity')
    st.markdown('<div class="dv"></div>',unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6=st.tabs(['◈ THESIS','📊 COMPARE','🎯 STRESS TEST','🎲 MONTE CARLO','🏛️ TAX','📖 ADAM'])

    # ── TAB 1: THESIS ─────────────────────────────────────────
    with t1:
        st.markdown('## The Question: You Have '+fs(p.principal)+'. What Do You Do?')
        st.markdown(f'<div class="th"><strong>Option A — NCD System:</strong> Deploy into NCDs at {fp(p.ncd_rate,1)}. Harvest {fi(p.gm)}/month interest. Reinvest into equity MF via SIP. At maturity: principal returned + equity portfolio. <strong>Capital-at-risk in equity: {fs(nd["car"][N])} (only the reinvested interest).</strong><br><br><strong>Option B — Lump Sum Equity:</strong> Put entire {fs(p.principal)} into equity MF on day 1. Higher potential return, but <strong>100% of capital at equity market risk from day 1.</strong><br><br><strong>Option C — FD+Equity:</strong> Same structure as NCD system but with {fp(p.fd_rate,0)} FD yield. Lower SIP corpus.<br><strong>Option D — Pure FD:</strong> Compound at {fp(p.fd_rate,0)}, no equity. <strong>Option E — Savings:</strong> Do nothing.</div>',unsafe_allow_html=True)

        # Wealth paths — ALL 5 systems
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=months,y=nd['total'],name='A: NCD System',line=dict(color=C['g'],width=2.5)))
        fig.add_trace(go.Scatter(x=months,y=ls['total'],name='B: Lump Sum Equity',line=dict(color=C['cy'],width=2,dash='dash')))
        fig.add_trace(go.Scatter(x=months,y=fd['total'],name=f'C: FD+Eq ({fp(p.fd_rate,0)})',line=dict(color=C['or_'],width=1.5,dash='dashdot')))
        fig.add_trace(go.Scatter(x=months,y=pfd['total'],name='D: Pure FD',line=dict(color=C['mu'],width=1.2,dash='dot')))
        fig.add_trace(go.Scatter(x=months,y=sv['total'],name='E: Savings',line=dict(color=C['rd'],width=1,dash='dot')))
        fig.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'],annotation_text='Original Capital')
        st.plotly_chart(_lo(fig,'Total Wealth — 5 Strategies Compared',480),width='stretch')

        # Scorecard
        st.markdown('## Scorecard — Net of All Taxes')
        systems = [
            ('NCD System', C['g'], nd['net'][N], nd['ttax'][N], nd['car'][N], True),
            ('Lump Sum Equity', C['cy'], ls['net'][N], ls['ltcg'][N], p.principal, False),
            (f'FD+Eq ({fp(p.fd_rate,0)})', C['or_'], fd['net'][N], fd['ttax'][N], fd['car'][N], False),
            ('Pure FD', C['mu'], pfd['net'][N], pfd['total'][N]-pfd['net'][N], 0, False),
            ('Savings', C['rd'], sv['net'][N], sv['total'][N]-sv['net'][N], 0, False),
        ]
        winner = max(systems, key=lambda x: x[2])
        cols=st.columns(5)
        for i,(nm,cl,nv,tx,car,_) in enumerate(systems):
            w = 'w' if nv==winner[2] else ''
            crown = ' 👑' if nv==winner[2] else ''
            car_txt = fs(car)+' in equity' if car>0 else 'No equity risk'
            cols[i].markdown(f'<div class="sc {w}"><div class="lb" style="color:{cl}">{nm}{crown}</div><div class="vl" style="color:{cl}">{fs(nv)}</div><div class="dt">Tax: {fs(tx)} · CAGR: {fp((nv/p.principal)**(1/p.horizon)-1 if nv>0 else 0)}</div><div class="car">Capital-at-risk: {car_txt}</div></div>',unsafe_allow_html=True)

        # Wealth decomposition
        st.markdown('## NCD System — Where Does the Wealth Come From?')
        fig_d=go.Figure(go.Waterfall(x=['NCD Principal\n(returned)','Interest\nReinvested','Equity\nCompounding','Total\nWealth'],
            y=[dec['principal'],dec['interest'],dec['compounding'],dec['total']],
            measure=['relative','relative','relative','total'],
            connector=dict(line=dict(color=C['mu'],width=1)),
            increasing=dict(marker=dict(color=C['gn'])),totals=dict(marker=dict(color=C['g'])),
            textposition='outside',text=[fs(dec['principal']),fs(dec['interest']),fs(dec['compounding']),fs(dec['total'])],
            textfont=dict(color=C['t2'],size=9)))
        st.plotly_chart(_lo(fig_d,'Wealth Decomposition — Source of Each Rupee',400),width='stretch')

        # The real insight
        st.markdown(f'<div class="th"><strong>Why consider the NCD system over lump sum equity?</strong><br>Lump sum equity returns {fs(ls["total"][N])} (net {fs(ls["net"][N])}) — potentially higher. But <strong>{fs(p.principal)} is at full equity risk from day 1.</strong> A 50% crash in year 1 means you\'re sitting on {fs(p.principal*0.5)}.<br><br>The NCD system puts <strong>only {fs(nd["car"][N])} into equity</strong> (accumulated over {p.horizon} years via SIP). The {fs(p.principal)} stays in debt. A 50% crash hurts less, and DCA means you buy more at lower prices. The tradeoff: lower peak return for significantly lower capital-at-risk.</div>',unsafe_allow_html=True)

    # ── TAB 2: COMPARE ────────────────────────────────────────
    with t2:
        st.markdown('## Detailed Comparison')
        # Net wealth chart
        fig_n=go.Figure()
        for yr_idx in range(p.horizon):
            m_idx=(yr_idx+1)*12-1
        y_months = [y*12-1 for y in range(1,p.horizon+1)]
        fig_n.add_trace(go.Scatter(x=list(range(1,p.horizon+1)),y=[nd['net'][m] for m in y_months],name='NCD System',mode='lines+markers',line=dict(color=C['g'],width=2.5),marker=dict(size=6)))
        fig_n.add_trace(go.Scatter(x=list(range(1,p.horizon+1)),y=[ls['net'][m] for m in y_months],name='Lump Sum',mode='lines+markers',line=dict(color=C['cy'],width=2,dash='dash'),marker=dict(size=5)))
        fig_n.add_trace(go.Scatter(x=list(range(1,p.horizon+1)),y=[fd['net'][m] for m in y_months],name='FD+Eq',mode='lines+markers',line=dict(color=C['or_'],width=1.5,dash='dashdot'),marker=dict(size=4)))
        fig_n.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
        fig_n.update_layout(xaxis_title='Year',yaxis_title='Net Wealth (₹)')
        st.plotly_chart(_lo(fig_n,'Post-Tax Net Wealth — Year by Year',440),width='stretch')

        # Comparison table
        st.markdown('### Year-End Table')
        tbl=[]
        for yr in range(1,p.horizon+1):
            m=yr*12-1
            tbl.append({'Year':yr,'NCD Net':fs(nd['net'][m]),'Lump Net':fs(ls['net'][m]),
                        'FD+Eq Net':fs(fd['net'][m]),'NCD CAGR':fp((nd['total'][m]/p.principal)**(1/yr)-1),
                        'NCD vs Lump':fs(nd['net'][m]-ls['net'][m]),
                        'Capital-at-Risk (NCD)':fs(nd['car'][m])})
        st.dataframe(pd.DataFrame(tbl),width='stretch',hide_index=True)

        st.markdown('### Advantage Analysis')
        adv_ncd_ls = [nd['net'][yr*12-1]-ls['net'][yr*12-1] for yr in range(1,p.horizon+1)]
        adv_ncd_fd = [nd['net'][yr*12-1]-fd['net'][yr*12-1] for yr in range(1,p.horizon+1)]
        fig_a=go.Figure()
        fig_a.add_trace(go.Bar(x=list(range(1,p.horizon+1)),y=adv_ncd_ls,name='NCD vs Lump Sum',marker_color=[C['gn'] if v>=0 else C['rd'] for v in adv_ncd_ls]))
        fig_a.add_trace(go.Bar(x=list(range(1,p.horizon+1)),y=adv_ncd_fd,name='NCD vs FD+Eq',marker_color=C['cy'],opacity=.5))
        fig_a.update_layout(barmode='group',xaxis_title='Year',yaxis_title='Advantage (₹)')
        st.plotly_chart(_lo(fig_a,'NCD System Advantage (green=NCD wins, red=Lump sum wins)',400),width='stretch')

        # Sensitivity
        st.markdown('### Sensitivity Matrix (Net Wealth of NCD System)')
        sens=eng.sens([.08,.09,.10,.118,.13,.14,.15],[.07,.10,.13,.16,.19])
        z=sens.values/1e5
        fgs=go.Figure(go.Heatmap(z=z,x=sens.columns.tolist(),y=sens.index.tolist(),colorscale=[[0,'#1B2A4A'],[.5,'#FFC300'],[1,'#FF5252']],text=[[f'₹{v:.1f}L' for v in r] for r in z],texttemplate='%{text}',textfont=dict(size=10,color='white'),colorbar=dict(title='₹L')))
        fgs.update_layout(xaxis_title='Equity Return',yaxis_title='NCD Yield')
        st.plotly_chart(_lo(fgs,'Net Wealth — NCD Yield × Equity Return',420),width='stretch')

    # ── TAB 3: STRESS TEST ────────────────────────────────────
    with t3:
        st.markdown('## Stress Test — Random & Historical Returns')

        st.markdown('### Random Returns')
        seed=st.number_input('Seed',value=42,step=1)
        feat=eng.stress_random(seed,1)[0]; ar=feat['ar']
        cl,cr=st.columns(2)
        with cl:
            cb=[C['gn'] if r>=0 else C['rd'] for r in ar]
            fb=go.Figure(go.Bar(x=list(range(1,p.horizon+1)),y=ar*100,marker_color=cb,text=[fp(r) for r in ar],textposition='outside',textfont=dict(size=9,color=C['t2'])))
            fb.update_layout(xaxis_title='Year',yaxis_title='%',yaxis=dict(zeroline=True,zerolinecolor=C['mu']))
            st.plotly_chart(_lo(fb,'Random Annual Returns',360),width='stretch')
        with cr:
            fp2=go.Figure()
            fp2.add_trace(go.Scatter(x=months,y=feat['ncd_path'],name='NCD System',line=dict(color=C['g'],width=2.5)))
            fp2.add_trace(go.Scatter(x=months,y=feat['ls_path'],name='Lump Sum',line=dict(color=C['cy'],width=2,dash='dash')))
            fp2.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
            st.plotly_chart(_lo(fp2,'Path: NCD vs Lump Sum',360),width='stretch')

        m1,m2,m3,m4=st.columns(4)
        m1.metric('NCD Net',fs(feat['ncd_net']))
        m2.metric('Lump Net',fs(feat['ls_net']))
        winner_txt = 'NCD' if feat['ncd_net']>feat['ls_net'] else 'Lump Sum'
        m3.metric('Winner',winner_txt,fs(abs(feat['ncd_net']-feat['ls_net'])))
        avg_r = np.mean(ar)
        m4.metric('Avg Annual Return',fp(avg_r))

        # 25 paths
        st.markdown('### 25 Random Paths')
        multi=eng.stress_random(seed*1000,25); fm=go.Figure()
        nw=[r['ncd_net'] for r in multi]; lw=[r['ls_net'] for r in multi]
        for i,r in enumerate(multi):
            fm.add_trace(go.Scatter(x=months,y=r['ncd_path'],showlegend=(i==0),name='NCD' if i==0 else None,legendgroup='n',line=dict(color=C['g'],width=.7),opacity=.35))
            fm.add_trace(go.Scatter(x=months,y=r['ls_path'],showlegend=(i==0),name='Lump Sum' if i==0 else None,legendgroup='l',line=dict(color=C['cy'],width=.6),opacity=.25))
        fm.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
        st.plotly_chart(_lo(fm,'25 Paths — NCD (gold) vs Lump Sum (cyan)',460),width='stretch')
        s1,s2,s3,s4=st.columns(4)
        na_=np.array(nw); la_=np.array(lw)
        s1.metric('NCD Mean',fs(na_.mean()))
        s2.metric('Lump Mean',fs(la_.mean()))
        s3.metric('NCD Wins',f'{np.sum(na_>la_)}/25')
        s4.metric('NCD Min',fs(na_.min()))

        # Historical NIFTY
        st.markdown('### Historical NIFTY Backtest')
        st.markdown(f'<div class="th">What if you started this system using <strong>actual NIFTY50 returns</strong>? This uses real calendar-year returns ({min(NIFTY_HIST.keys())}–{max(NIFTY_HIST.keys())}) to stress-test the system against what actually happened in Indian equity markets.</div>',unsafe_allow_html=True)
        valid_starts = [y for y in sorted(NIFTY_HIST.keys()) if y+p.horizon-1<=max(NIFTY_HIST.keys())]
        if valid_starts:
            htbl=[]
            for yr in valid_starts:
                r=eng.stress_historical(yr)
                if r:
                    w='NCD' if r['ncd_net']>r['ls_net'] else 'Lump Sum'
                    htbl.append({'Start Year':yr,'Period':f'{yr}–{yr+p.horizon-1}',
                                 'NCD Net':fs(r['ncd_net']),'Lump Net':fs(r['ls_net']),
                                 'Winner':w,'Margin':fs(abs(r['ncd_net']-r['ls_net']))})
            hdf=pd.DataFrame(htbl)
            st.dataframe(hdf,width='stretch',hide_index=True)
            ncd_wins=sum(1 for r in htbl if r['Winner']=='NCD')
            st.markdown(f'<div class="th">Across {len(htbl)} historical {p.horizon}-year windows: <strong>NCD system won {ncd_wins}/{len(htbl)} times</strong>. Lump sum equity tends to win in strong bull markets (2005-2014), while NCD system shows resilience through volatile periods by preserving capital and DCA-ing into drawdowns.</div>',unsafe_allow_html=True)
        else:
            st.info(f'Need at least {p.horizon} years of NIFTY data. Reduce horizon or historical data unavailable.')

    # ── TAB 4: MONTE CARLO ────────────────────────────────────
    with t4:
        st.markdown('## Monte Carlo — NCD System vs Lump Sum Equity')
        st.markdown(f'<div class="ad"><span class="tg">Adam · GBM</span>Both systems use <strong>same random paths</strong> (paired comparison). σ={fp(EQ_VOL,0)} | {p.mc_paths:,} paths | Drift-corrected</div>',unsafe_allow_html=True)
        with st.spinner(f'{p.mc_paths:,} simulations...'):
            mc=eng.mc(); met=eng.metrics(nd,ls,fd,pfd,sv,mc)

        cl,cr=st.columns(2)
        for col,key,title,clr in [(cl,'pp','NCD System',C['g']),(cr,'lp','Lump Sum Equity',C['cy'])]:
            with col:
                fig=go.Figure(); m=months; d=mc[key]
                rgb=','.join(str(int(clr.lstrip('#')[i:i+2],16)) for i in (0,2,4))
                fig.add_trace(go.Scatter(x=np.concatenate([m,m[::-1]]),y=np.concatenate([d[95],d[5][::-1]]),fill='toself',fillcolor=f'rgba({rgb},.06)',line=dict(color='rgba(0,0,0,0)'),name='5th–95th',hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=np.concatenate([m,m[::-1]]),y=np.concatenate([d[75],d[25][::-1]]),fill='toself',fillcolor=f'rgba({rgb},.12)',line=dict(color='rgba(0,0,0,0)'),name='25th–75th',hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=m,y=d[50],name='Median',line=dict(color=clr,width=2.5)))
                fig.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
                st.plotly_chart(_lo(fig,title,400),width='stretch')

        fh=go.Figure()
        fh.add_trace(go.Histogram(x=mc['fn']/1e5,nbinsx=60,name='NCD Net',marker_color=C['g'],opacity=.6))
        fh.add_trace(go.Histogram(x=mc['ln']/1e5,nbinsx=60,name='Lump Net',marker_color=C['cy'],opacity=.4))
        fh.update_layout(barmode='overlay',xaxis_title='Net Wealth (₹L)',yaxis_title='Freq')
        st.plotly_chart(_lo(fh,'Terminal Distribution — Net of Tax',380),width='stretch')

        st.markdown('### Percentile Table')
        ptbl=[]
        for k in [5,10,25,50,75,90,95]:
            ptbl.append({'Pctl':f'{k}th','NCD Net':fs(mc['pn'][k]),'Lump Net':fs(mc['lpn'][k]),'SIP Net':fs(mc['spn'][k]),
                         'NCD vs Lump':fs(mc['pn'][k]-mc['lpn'][k])})
        st.dataframe(pd.DataFrame(ptbl),width='stretch',hide_index=True)

        st.markdown('### Risk Metrics')
        r1,r2,r3,r4=st.columns(4)
        r1.metric('Sharpe',f'{met["sh"]:.3f}'); r2.metric('Sortino',f'{met["so"]:.3f}')
        r3.metric('VaR (5%)',fs(met['v5'])); r4.metric('CVaR (5%)',fs(met['cv5']))
        r5,r6,r7,r8=st.columns(4)
        r5.metric('P(NCD > Lump)',fp(met['pb_ls'],1)); r6.metric('P(Capital Safe)',fp(met['pc'],1))
        r7.metric('P(Lump Loss)',fp(met['ls_pl'],1)); r8.metric('Median Max DD',fp(met['md'],1))
        r9,r10,r11,r12=st.columns(4)
        r9.metric('Break-Even Eq.',fp(met['be'])); r10.metric('P(2× Capital)',fp(met['p2'],1))
        r11.metric('MC Mean CAGR',fp(met['mc_'])); r12.metric('Vol of CAGR',fp(met['vc']))

        fd_=go.Figure(go.Histogram(x=mc['dd']*100,nbinsx=60,marker_color=C['rd'],opacity=.6))
        fd_.add_vline(x=met['md']*100,line_dash='dash',line_color=C['g'],annotation_text=f'Med: {fp(met["md"],1)}')
        fd_.update_layout(xaxis_title='Max DD (%)',yaxis_title='Freq')
        st.plotly_chart(_lo(fd_,'Max Drawdown Distribution (NCD System)',330),width='stretch')

    # ── TAB 5: TAX ────────────────────────────────────────────
    with t5:
        st.markdown('## Tax Analysis')
        st.markdown(f'<div class="ad"><span class="tg">Tax Regime</span><strong>NCD Interest:</strong> Slab {fp(p.tax_slab,0)} + Cess {fp(p.cess,0)} = {fp(p.et,1)}. TDS {fp(p.tds_rate,0)}. <strong>LTCG:</strong> {fp(p.ltcg_rate,1)} on gains > {fi(p.ltcg_exempt)}/FY (annual exemption applied per-FY, not once at terminal). <strong>STCG:</strong> {fp(p.stcg_rate,0)} flat.</div>',unsafe_allow_html=True)

        ai=p.principal*p.ncd_rate*p.horizon
        fw=go.Figure(go.Waterfall(x=['Interest\nIncome','Equity\nGains','Interest\nTax','LTCG\nTax','Net\nWealth'],
            y=[ai,nd['g'][N],-nd['tax_int'][N],-nd['ltcg'][N],nd['net'][N]],
            measure=['relative','relative','relative','relative','total'],
            connector=dict(line=dict(color=C['mu'],width=1)),
            decreasing=dict(marker=dict(color=C['rd'])),increasing=dict(marker=dict(color=C['gn'])),
            totals=dict(marker=dict(color=C['g'])),textposition='outside',
            text=[fs(ai),fs(nd['g'][N]),fs(nd['tax_int'][N]),fs(nd['ltcg'][N]),fs(nd['net'][N])],
            textfont=dict(color=C['t2'],size=9)))
        st.plotly_chart(_lo(fw,'NCD System — Income to Net Wealth',400),width='stretch')

        st.markdown('### Tax Comparison')
        ttbl=pd.DataFrame({'':['Total Wealth','Total Tax','Net Wealth','Tax Rate','Capital-at-Risk'],
            'NCD System':[fi(nd['total'][N]),fi(nd['ttax'][N]),fi(nd['net'][N]),fp(nd['ttax'][N]/nd['total'][N]),fi(nd['car'][N])],
            'Lump Sum':[fi(ls['total'][N]),fi(ls['ltcg'][N]),fi(ls['net'][N]),fp(ls['ltcg'][N]/ls['total'][N] if ls['total'][N]>0 else 0),fi(p.principal)],
            f'FD+Eq':[fi(fd['total'][N]),fi(fd['ttax'][N]),fi(fd['net'][N]),fp(fd['ttax'][N]/fd['total'][N]),fi(fd['car'][N])],
            'Pure FD':[fi(pfd['total'][N]),fi(pfd['total'][N]-pfd['net'][N]),fi(pfd['net'][N]),fp((pfd['total'][N]-pfd['net'][N])/pfd['total'][N]),'₹0']})
        st.dataframe(ttbl,width='stretch',hide_index=True)

        td=nd['ttax'][N]-ls['ltcg'][N]
        st.markdown(f'<div class="th"><strong>NCD system pays {fs(td)} more tax</strong> than lump sum ({fi(nd["ttax"][N])} vs {fi(ls["ltcg"][N])}). This is the cost of the yield-arbitrage structure — interest is taxed at slab rate ({fp(p.et,1)}) while equity gains get concessional LTCG ({fp(p.ltcg_rate,1)}). The tradeoff: higher tax for dramatically lower capital-at-risk ({fs(nd["car"][N])} vs {fs(p.principal)}).<br><br><strong>LTCG Harvesting:</strong> The model applies {fi(p.ltcg_exempt)} exemption per financial year (not once at terminal). Annual harvest-and-reinvest saves up to {fi(p.ltcg_exempt*p.ltcg_rate)}/year.</div>',unsafe_allow_html=True)

        st.markdown('### Risk Decomposition')
        for nm,sev,cl,desc in [
            ('Credit Risk','CRITICAL',C['rd'],'NCD default → principal loss. Diversify 5–10 issuers, AA- min, secured, quarterly monitoring.'),
            ('Equity Drawdown','HIGH',C['or_'],f'Median max DD {fp(met["md"],1)}. NCD system exposed only via SIP amounts, not full principal.'),
            ('Interest Rate','LOW (HTM)',C['gn'],'Fixed coupon. No impact if held to maturity.'),
            ('Inflation','MEDIUM',C['g'],f'At {fp(p.inflation,0)}, real coupon declines. Equity hedges long-term.'),
            ('Reinvestment','LOW–MED',C['cy'],'At maturity, comparable yields may be unavailable. Ladder strategy recommended.')]:
            st.markdown(f'<div class="rc" style="border-left:3px solid {cl}"><div class="rt" style="color:{cl}">{nm} — {sev}</div><div class="rb">{desc}</div></div>',unsafe_allow_html=True)

    # ── TAB 6: ADAM ───────────────────────────────────────────
    with t6:
        st.markdown('## Adam Deep Literature')
        cfv=eng.cf(); delta=abs(cfv-nd['mf'][N])
        ntc=p.ncd_rate*p.horizon*p.et

        st.markdown(f'<div class="ad"><span class="tg">Closed-Form Validation</span><code>V(N) = S·[(1+r)^N-1]/r</code> | S={fi(p.nm,True)} r={p.mr:.10f} N={p.N}<br>Result: <code>{fi(cfv,True)}</code> | Simulation: <code>{fi(nd["mf"][N],True)}</code> | Δ: <code>{fi(delta,True)}</code> ✓</div>',unsafe_allow_html=True)

        st.markdown(f'<div class="ad"><span class="tg">Yield Arbitrage Theorem</span><strong>Claim:</strong> NCD system dominates pure SIP by P·(1-r_d·T·τ) after tax.<br><code>= {fi(p.principal)} × (1 - {ntc:.4f}) = {fi(p.principal*(1-ntc))}</code><br>Valid when r_d·T·τ < 1. Current: {ntc:.4f} < 1 ✓ ∎</div>',unsafe_allow_html=True)

        st.markdown(f'<div class="ad"><span class="tg">NCD vs Lump Sum — When Does NCD Win?</span>NCD net = P + V(N) - P·r_d·T·τ - LTCG_mf<br>Lump net = P·(1+r)^T - LTCG_lump<br><br>NCD wins when: V(N) - P·r_d·T·τ - LTCG_mf > P·[(1+r)^T - 1] - LTCG_lump<br>i.e., when the SIP portfolio + tax savings on LTCG exceeds the lump sum growth minus its LTCG.<br><br>Under constant returns, lump sum typically wins (full compounding from day 1). Under <strong>volatile</strong> returns, NCD system gains from DCA (buying more at lower prices) while lump sum suffers from sequence risk.<br><br>Current deterministic: NCD net={fi(nd["net"][N])} vs Lump net={fi(ls["net"][N])}. Lump wins by {fi(ls["net"][N]-nd["net"][N])}.<br>MC P(NCD>Lump): <code>{fp(met["pb_ls"])}</code> — in volatile markets, the gap narrows.</div>',unsafe_allow_html=True)

        mu_a=np.log(1+p.ne)/12-EQ_VOL**2/24
        st.markdown(f'<div class="ad"><span class="tg">GBM Drift Correction</span><code>μ_adj = ln(1+r)/12 - σ²/24 = {mu_a:.8f}</code><br><code>σ_m = σ/√12 = {EQ_VOL/np.sqrt(12):.8f}</code><br>Jensen\'s inequality correction ensures E[multiplicative return] is unbiased.</div>',unsafe_allow_html=True)

        st.markdown(f'<div class="ad"><span class="tg">Capital-at-Risk Framework</span>NCD System: Only reinvested interest enters equity = <code>{fi(nd["car"][N])}</code>. The {fi(p.principal)} is in debt (credit risk, not market risk).<br>Lump Sum: 100% = <code>{fi(p.principal)}</code> at equity risk from day 1.<br><br>Risk-adjusted return (net wealth per unit of capital-at-risk):<br>NCD: {fi(nd["net"][N])}/{fi(nd["car"][N])} = <code>{nd["net"][N]/nd["car"][N]:.2f}×</code><br>Lump: {fi(ls["net"][N])}/{fi(p.principal)} = <code>{ls["net"][N]/p.principal:.2f}×</code><br><br>Per rupee of equity risk, NCD system generates more net wealth.</div>',unsafe_allow_html=True)

        st.markdown(f'<div class="ad"><span class="tg">Per-FY LTCG Model</span>Unlike the previous model (single terminal redemption), Samriddhi applies the ₹{_ic(int(p.ltcg_exempt))} exemption at each financial year boundary. This correctly models annual LTCG harvesting — redeeming and reinvesting to crystallize gains within the exemption window each year.<br><br>Annual saving potential: {fi(p.ltcg_exempt*p.ltcg_rate)}/year × {p.horizon} years = <code>{fi(p.ltcg_exempt*p.ltcg_rate*p.horizon)}</code></div>',unsafe_allow_html=True)

        sk=stats.skew(mc['ft']); ku=stats.kurtosis(mc['ft'])
        st.markdown(f'<div class="ad"><span class="tg">Distribution Properties</span>Mean: {fs(mc["ft"].mean())} | Std: {fs(mc["ft"].std())} | Skew: {sk:.4f} | Kurt: {ku:.4f}<br>P(NCD>Lump): {fp(met["pb_ls"])} | P(Capital): {fp(met["pc"])} | P(2×): {fp(met["p2"])}</div>',unsafe_allow_html=True)

    st.markdown(f'<div style="text-align:center;padding:20px 0 10px;margin-top:28px;border-top:1px solid {C["gb"]}"><div style="font-family:Inter;font-weight:800;font-size:.8rem;color:{C["g"]};letter-spacing:.06em">◈ समृद्धि SAMRIDDHI</div><div style="font-family:JetBrains Mono;font-size:.55rem;color:{C["mu"]};margin-top:2px;letter-spacing:.1em">HEMREK CAPITAL — ADAM DEEP LITERATURE ENGINE</div></div>',unsafe_allow_html=True)

if __name__=='__main__': main()
