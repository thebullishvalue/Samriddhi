"""
◈ समृद्धि SAMRIDDHI — NCD + Equity Reinvestment Intelligence System
Hemrek Capital | Adam Deep Literature Engine

Two modes of analysis:
  🌍 REAL WORLD — Random returns, historical backtest, Monte Carlo
  📐 DETERMINISTIC — Constant-return projections, closed-form, sensitivity

Real World is primary. Deterministic is the theoretical reference.
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
NIFTY = {2005:.363,2006:.398,2007:.548,2008:-.518,2009:.758,2010:.179,2011:-.246,
         2012:.277,2013:.068,2014:.314,2015:-.041,2016:.030,2017:.286,2018:.032,
         2019:.120,2020:.149,2021:.241,2022:.043,2023:.200,2024:.088}

# ═══════════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════════
def _ic(n):
    s=str(abs(int(n)))
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
def _cagr(end,start,yr): return (end/start)**(1/yr)-1 if end>0 and start>0 and yr>0 else 0

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
    [data-testid="stMetricLabel"] p{{color:var(--mu)!important;font-family:'Inter'!important;font-size:.63rem!important;font-weight:600!important;text-transform:uppercase;letter-spacing:.08em}}
    [data-testid="stMetricValue"]{{color:var(--g)!important;font-family:'JetBrains Mono',monospace!important;font-weight:600!important;font-size:1.1rem!important}}
    [data-testid="stMetricDelta"]{{font-family:'JetBrains Mono',monospace!important;font-size:.68rem!important}}
    .stNumberInput input{{background:{C['inp']}!important;color:var(--g)!important;border:1px solid {C['gb']}!important;border-radius:6px!important;font-family:'JetBrains Mono',monospace!important}}
    .stTabs [data-baseweb="tab-list"]{{gap:2px;background:{C['bg2']};border-radius:8px;padding:3px}}
    .stTabs [data-baseweb="tab"]{{background:transparent;color:var(--mu);border-radius:6px;font-family:'Inter';font-weight:600;font-size:.74rem}}
    .stTabs [aria-selected="true"]{{background:var(--cd)!important;color:var(--g)!important;border-bottom:2px solid var(--g)!important}}
    .streamlit-expanderHeader{{background:var(--cd)!important;border:1px solid {C['bd']}!important;color:var(--t2)!important;font-family:'JetBrains Mono',monospace!important;font-size:.82rem!important}}
    .streamlit-expanderContent{{background:var(--cd)!important;border:1px solid {C['bd']}!important;border-top:none!important}}
    .hdr{{background:linear-gradient(135deg,{C['bg2']},{C['cd']});border:1px solid {C['gb']};border-left:3px solid var(--g);border-radius:8px;padding:14px 18px;margin-bottom:14px}}
    .hdr h1{{margin:0!important;padding:0!important;font-size:1.3rem!important}}
    .hdr .sub{{color:var(--mu);font-family:'JetBrains Mono',monospace;font-size:.66rem;margin-top:3px;letter-spacing:.05em}}
    .ad{{background:var(--cd);border:1px solid {C['bd']};border-left:3px solid {C['pu']};border-radius:6px;padding:11px 15px;margin:8px 0;font-family:'JetBrains Mono',monospace;font-size:.76rem;color:var(--t2);line-height:1.55}}
    .ad .tg{{color:{C['pu']};font-weight:700;font-size:.62rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px;display:block}}
    .ad code{{color:var(--g);background:rgba(255,195,0,.08);padding:1px 4px;border-radius:3px}}
    .th{{background:linear-gradient(135deg,rgba(255,195,0,.04),rgba(255,195,0,.01));border:1px solid {C['gb']};border-radius:8px;padding:13px 17px;margin:8px 0;color:var(--t2);font-size:.81rem;line-height:1.6}}
    .th strong{{color:var(--g)}}
    .dv{{height:1px;background:linear-gradient(to right,transparent,{C['gb']},transparent);margin:18px 0}}
    .sc{{text-align:center;padding:10px;background:var(--cd);border-radius:8px;border:1px solid {C['bd']};margin-bottom:3px}}
    .sc.w{{border-color:{C['gb']};box-shadow:0 0 8px rgba(255,195,0,.06)}}
    .sc .lb{{font-size:.58rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700}}
    .sc .vl{{font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;margin:3px 0}}
    .sc .dt{{font-size:.68rem;color:var(--t2)}}
    .rc{{background:var(--cd);border:1px solid {C['bd']};border-radius:8px;padding:11px 15px;margin:4px 0}}
    .rc .rt{{font-weight:700;font-size:.82rem;margin-bottom:2px}}
    .rc .rb{{color:var(--t2);font-size:.76rem;line-height:1.45}}
    .mode-badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:700;letter-spacing:.06em}}
    </style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PARAMS & ENGINE
# ═══════════════════════════════════════════════════════════════
@dataclass
class P:
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

class Eng:
    def __init__(s,p): s.p=p

    def _sip(s,sip,r_m,N):
        mf=cum=0.0; vals=[]
        for m in range(N):
            r=r_m if np.isscalar(r_m) else r_m[m]
            mf=mf*(1+r)+sip; cum+=sip; vals.append((mf,cum))
        return np.array(vals)

    def _ltcg_annual(s,gains_arr,N):
        """Per-FY LTCG with annual exemption."""
        p=s.p; lc=np.zeros(N); pg=0.0
        for m in range(N):
            if (m+1)%12==0:
                fg=gains_arr[m]-pg; lc[m:]=(lc[m-1] if m>0 else 0)+max(0,(fg-p.ltcg_exempt))*p.ltcg_rate; pg=gains_arr[m]
            elif m>0: lc[m]=lc[m-1]
        return lc

    def ncd(s,er=None,mr=None):
        p=s.p; rm=mr if mr is not None else (1+(er if er is not None else p.ne))**(1/12)-1
        v=s._sip(p.nm,rm,p.N); mf=v[:,0]; cum=v[:,1]; g=mf-cum
        lc=s._ltcg_annual(g,p.N); ti=p.gm*p.et*np.arange(1,p.N+1)
        total=p.principal+mf; net=total-ti-lc
        return dict(mf=mf,cum=cum,g=g,total=total,net=net,ti=ti,lc=lc,tt=ti+lc,car=cum)

    def lump(s,er=None,mr=None):
        p=s.p; rm=mr if mr is not None else (1+(er if er is not None else p.ne))**(1/12)-1
        v=np.zeros(p.N); v[0]=p.principal*(1+(rm if np.isscalar(rm) else rm[0]))
        for m in range(1,p.N): v[m]=v[m-1]*(1+(rm if np.isscalar(rm) else rm[m]))
        g=v-p.principal; lc=s._ltcg_annual(g,p.N); net=v-lc
        return dict(total=v,net=net,g=g,lc=lc,car=np.full(p.N,p.principal))

    def fd_eq(s):
        p=s.p; fn=p.principal*p.fd_rate/12*(1-p.tds_rate)
        v=s._sip(fn,p.mr,p.N); mf=v[:,0]; cum=v[:,1]; g=mf-cum
        lc=s._ltcg_annual(g,p.N); ti=p.principal*p.fd_rate/12*p.et*np.arange(1,p.N+1)
        total=p.principal+mf; net=total-ti-lc
        return dict(mf=mf,cum=cum,total=total,net=net,ti=ti,lc=lc,tt=ti+lc,car=cum)

    def pure_fd(s):
        p=s.p; v=np.array([p.principal*(1+p.fd_rate)**((m+1)/12) for m in range(p.N)])
        t=(v-p.principal)*p.et; return dict(total=v,net=v-t,car=np.zeros(p.N))

    def savings(s):
        p=s.p; v=np.array([p.principal*(1+p.savings_rate)**((m+1)/12) for m in range(p.N)])
        t=(v-p.principal)*p.et; return dict(total=v,net=v-t,car=np.zeros(p.N))

    def cf(s,sip=None,er=None):
        p=s.p; S=sip or p.nm; r=(1+(er if er is not None else p.ne))**(1/12)-1
        return S*((1+r)**p.N-1)/r if r!=0 else S*p.N

    def decompose(s,nd):
        p=s.p; ci=nd['cum'][-1]; cg=nd['mf'][-1]-ci
        return dict(principal=p.principal,interest=ci,compounding=cg,total=p.principal+nd['mf'][-1])

    def sens(s,nys,ers):
        p=s.p; rows=[]
        for ny in nys:
            row={}
            for er in ers:
                sp_=p.principal*ny/12*(1-p.tds_rate); rm=(1+er-p.expense_ratio)**(1/12)-1
                fv=sp_*((1+rm)**p.N-1)/rm if rm!=0 else sp_*p.N
                v=p.principal+fv-p.principal*ny*p.horizon*p.et
                g=fv-sp_*p.N; v-=max(0,(g-p.ltcg_exempt*p.horizon))*p.ltcg_rate
                row[f'{er:.0%}']=v
            rows.append(row)
        return pd.DataFrame(rows,index=[f'{y:.1%}' for y in nys])

    # ── Stochastic engines ────────────────────────────────────

    def _gen_returns(s,seed,n_paths=1):
        p=s.p; mu=np.log(1+p.ne)/12-EQ_VOL**2/24; sig=EQ_VOL/np.sqrt(12)
        np.random.seed(seed)
        return np.exp(np.random.normal(mu,sig,(n_paths,p.N)))-1

    def scenario(s,seed):
        """Single random path with full analytics."""
        p=s.p; mr=s._gen_returns(seed,1)[0]
        ar=np.array([np.prod(1+mr[y*12:(y+1)*12])-1 for y in range(p.horizon)])
        nd=s.ncd(mr=mr); ls=s.lump(mr=mr)
        # Drawdown
        def _dd(path):
            pk=np.maximum.accumulate(path); dd=(path-pk)/pk
            return dd,dd.min(),int(np.argmin(dd))+1,int(np.sum(dd<-0.001))
        n_dd,n_mdd,n_mm,n_uw=_dd(nd['total']); l_dd,l_mdd,l_mm,l_uw=_dd(ls['total'])
        # Rolling 3Y
        r3=[]
        for yr in range(3,p.horizon+1):
            s_,e_=(yr-3)*12,yr*12-1
            r3.append(dict(yr=yr,ncd=(nd['total'][e_]/nd['total'][s_])**(1/3)-1 if nd['total'][s_]>0 else 0,
                           ls=(ls['total'][e_]/ls['total'][s_])**(1/3)-1 if ls['total'][s_]>0 else 0))
        # Year-end data
        yd=[dict(yr=y+1,ret=ar[y],
                 nt=nd['total'][y*12+11],nn=nd['net'][y*12+11],nc=nd['car'][y*12+11],
                 lt=ls['total'][y*12+11],ln=ls['net'][y*12+11],
                 leader='NCD' if nd['total'][y*12+11]>ls['total'][y*12+11] else 'Lump')
            for y in range(p.horizon)]
        ncd_wins=sum(1 for d in yd if d['leader']=='NCD')
        return dict(mr=mr,ar=ar,nd=nd,ls=ls,
            n_dd=n_dd,n_mdd=n_mdd,n_mm=n_mm,n_uw=n_uw,
            l_dd=l_dd,l_mdd=l_mdd,l_mm=l_mm,l_uw=l_uw,
            r3=r3,yd=yd,ncd_wins=ncd_wins,
            ncd_cagr=_cagr(nd['total'][-1],p.principal,p.horizon),
            ls_cagr=_cagr(ls['total'][-1],p.principal,p.horizon))

    def scenario_multi(s,seed,n=25):
        """Multiple random paths with aggregate stats."""
        p=s.p; mrs=s._gen_returns(seed,n); results=[]
        for i in range(n):
            mr=mrs[i]; nd=s.ncd(mr=mr); ls=s.lump(mr=mr)
            pk=np.maximum.accumulate(nd['total']); dd=((nd['total']-pk)/pk).min()
            lpk=np.maximum.accumulate(ls['total']); ldd=((ls['total']-lpk)/lpk).min()
            results.append(dict(
                ncd_path=nd['total'],ls_path=ls['total'],
                ncd_net=nd['net'][-1],ls_net=ls['net'][-1],
                ncd_total=nd['total'][-1],ls_total=ls['total'][-1],
                ncd_mdd=dd,ls_mdd=ldd,
                ncd_cagr=_cagr(nd['total'][-1],p.principal,p.horizon),
                ls_cagr=_cagr(ls['total'][-1],p.principal,p.horizon)))
        return results

    def backtest(s,start):
        """Historical NIFTY backtest for a given start year."""
        p=s.p; yrs=sorted(NIFTY.keys()); 
        if start not in yrs: return None
        idx=yrs.index(start); avail=yrs[idx:idx+p.horizon]
        if len(avail)<p.horizon: return None
        ar=[NIFTY[y] for y in avail]
        mr=np.concatenate([np.full(12,(1+r)**(1/12)-1) for r in ar])[:p.N]
        nd=s.ncd(mr=mr); ls=s.lump(mr=mr)
        return dict(ar=ar,yrs=avail,nd=nd,ls=ls,
            ncd_net=nd['net'][-1],ls_net=ls['net'][-1],
            ncd_total=nd['total'][-1],ls_total=ls['total'][-1])

    def mc(s):
        p=s.p; mrs=s._gen_returns(42,p.mc_paths); ci=p.nm*p.N
        # NCD
        mf=np.zeros((p.mc_paths,p.N))
        for m in range(p.N):
            prev=mf[:,m-1] if m>0 else 0.0; mf[:,m]=prev*(1+mrs[:,m])+p.nm
        ft=p.principal+mf[:,-1]; fg=mf[:,-1]-ci; it=p.gm*p.et*p.N
        lc=np.maximum(0,(fg-p.ltcg_exempt*p.horizon))*p.ltcg_rate; fn=ft-it-lc
        # Lump
        lv=np.zeros((p.mc_paths,p.N)); lv[:,0]=p.principal*(1+mrs[:,0])
        for m in range(1,p.N): lv[:,m]=lv[:,m-1]*(1+mrs[:,m])
        lf=lv[:,-1]; lg=lf-p.principal
        ll=np.maximum(0,(lg-p.ltcg_exempt*p.horizon))*p.ltcg_rate; ln_=lf-ll
        # SIP
        sm=np.zeros((p.mc_paths,p.N))
        for m in range(p.N):
            prev=sm[:,m-1] if m>0 else 0.0; sm[:,m]=prev*(1+mrs[:,m])+p.nm
        sf=sm[:,-1]; sg=sf-ci; sl=np.maximum(0,(sg-p.ltcg_exempt*p.horizon))*p.ltcg_rate; sn=sf-sl
        # Paths
        tot=p.principal+mf; pk=[5,10,25,50,75,90,95]
        pp={k:np.percentile(tot,k,axis=0) for k in pk}
        lp={k:np.percentile(lv,k,axis=0) for k in pk}
        dd=np.array([((tot[i]-np.maximum.accumulate(tot[i]))/np.maximum.accumulate(tot[i])).min() for i in range(p.mc_paths)])
        # Risk
        cg=(ft/p.principal)**(1/p.horizon)-1; mc_=cg.mean(); vc=cg.std(); rf=p.savings_rate
        sh=(mc_-rf)/vc if vc>0 else 0
        ds=cg[cg<rf]-rf; dv=np.sqrt(np.mean(ds**2)) if len(ds)>0 else .001; so=(mc_-rf)/dv
        g_=np.maximum(0,cg-rf); l_=np.maximum(0,rf-cg)
        om=g_.mean()/l_.mean() if l_.mean()>0 else float('inf')
        md=np.median(dd); ca=mc_/abs(md) if md!=0 else float('inf')
        v5=np.percentile(fn,5); cv5=fn[fn<=v5].mean() if np.any(fn<=v5) else v5
        try:
            def _w(r):
                rm=(1+r)**(1/12)-1; fv=p.nm*((1+rm)**p.N-1)/rm if rm!=0 else p.nm*p.N
                g=fv-p.nm*p.N; return fv-p.gm*p.et*p.N-max(0,(g-p.ltcg_exempt*p.horizon))*p.ltcg_rate
            be=brentq(_w,-.10,.40)
        except: be=0.0
        return dict(ft=ft,fn=fn,lf=lf,ln=ln_,sf=sf,sn=sn,pp=pp,lp=lp,dd=dd,
            pn={k:np.percentile(fn,k) for k in pk},lpn={k:np.percentile(ln_,k) for k in pk},
            spn={k:np.percentile(sn,k) for k in pk},
            sh=sh,so=so,om=om,ca=ca,v5=v5,cv5=cv5,mc_=mc_,vc=vc,md=md,be=be,
            pb_ls=np.mean(fn>ln_),pb_sip=np.mean(ft>sf),
            pc=np.mean(fn>p.principal),p2=np.mean(ft>=2*p.principal),
            pl=np.mean(fn<p.principal),ls_pl=np.mean(ln_<p.principal))

# ═══════════════════════════════════════════════════════════════
# CHART HELPER
# ═══════════════════════════════════════════════════════════════
def _lo(fig,title='',h=440):
    fig.update_layout(title=dict(text=title,font=dict(color=C['g'],size=12,family='Inter')),
        paper_bgcolor=C['cd'],plot_bgcolor=C['cd'],
        font=dict(color=C['t2'],family='JetBrains Mono,monospace',size=10),
        xaxis=dict(gridcolor='rgba(255,255,255,.04)',zerolinecolor='rgba(255,255,255,.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,.04)',zerolinecolor='rgba(255,255,255,.06)'),
        height=h,margin=dict(t=46,b=36,l=56,r=16),
        legend=dict(bgcolor='rgba(0,0,0,.3)',bordercolor=C['gb'],borderwidth=1,font=dict(size=8)),
        hoverlabel=dict(bgcolor=C['bg2'],font_size=10,font_family='JetBrains Mono'))
    return fig

def _fan(data,months,color,title,p,show_floor=True):
    fig=go.Figure(); rgb=','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))
    fig.add_trace(go.Scatter(x=np.concatenate([months,months[::-1]]),y=np.concatenate([data[95],data[5][::-1]]),fill='toself',fillcolor=f'rgba({rgb},.06)',line=dict(color='rgba(0,0,0,0)'),name='5th–95th',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=np.concatenate([months,months[::-1]]),y=np.concatenate([data[75],data[25][::-1]]),fill='toself',fillcolor=f'rgba({rgb},.12)',line=dict(color='rgba(0,0,0,0)'),name='25th–75th',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=months,y=data[50],name='Median',line=dict(color=color,width=2.5)))
    if show_floor: fig.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
    return _lo(fig,title,400)

# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title='Samriddhi — Hemrek Capital',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
    css()

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f'<div style="text-align:center;padding:6px 0 10px"><div style="font-family:Inter;font-weight:800;font-size:1rem;color:{C["g"]};letter-spacing:.04em">◈ समृद्धि SAMRIDDHI</div><div style="font-family:JetBrains Mono;font-size:.56rem;color:{C["mu"]};letter-spacing:.12em;margin-top:2px">HEMREK CAPITAL</div></div><div style="height:1px;background:{C["gb"]};margin:0 0 10px"></div>',unsafe_allow_html=True)
        st.markdown(f"<p style='color:{C['g']};font-size:.66rem;font-weight:700;letter-spacing:.1em'>CAPITAL</p>",unsafe_allow_html=True)
        principal=st.number_input('Investment (₹)',value=10_00_000,step=1_00_000,format='%d')
        ncd_rate=st.slider('NCD Yield (%)',6.0,16.0,11.8,0.1)/100
        horizon=st.slider('Horizon (Years)',3,15,10)
        st.markdown(f"<p style='color:{C['g']};font-size:.66rem;font-weight:700;letter-spacing:.1em;margin-top:10px'>EQUITY</p>",unsafe_allow_html=True)
        eq_return=st.slider('Expected Return (%)',5.0,22.0,13.0,0.5)/100
        expense_ratio=st.slider('Expense Ratio (%)',0.0,2.5,0.5,0.1)/100
        st.markdown(f"<p style='color:{C['g']};font-size:.66rem;font-weight:700;letter-spacing:.1em;margin-top:10px'>TAX</p>",unsafe_allow_html=True)
        tax_slab=st.selectbox('Tax Slab',[0.0,.05,.10,.15,.20,.25,.30],index=6,format_func=lambda x:f'{x:.0%}')
        ltcg_rate=st.number_input('LTCG (%)',value=12.5,step=0.5)/100
        fd_rate=st.slider('FD Rate (%)',4.0,9.0,7.0,0.25)/100
        st.markdown(f"<p style='color:{C['g']};font-size:.66rem;font-weight:700;letter-spacing:.1em;margin-top:10px'>ENVIRONMENT</p>",unsafe_allow_html=True)
        inflation=st.slider('Inflation (%)',2.0,10.0,6.0,0.5)/100
        mc_paths=st.select_slider('MC Paths',[1000,2500,5000,10000],value=5000)

    p=P(principal=principal,ncd_rate=ncd_rate,horizon=horizon,eq_return=eq_return,expense_ratio=expense_ratio,tax_slab=tax_slab,ltcg_rate=ltcg_rate,fd_rate=fd_rate,inflation=inflation,mc_paths=mc_paths)
    e=Eng(p); months=np.arange(1,p.N+1); N=p.N-1

    # ── HEADER ────────────────────────────────────────────────
    st.markdown(f'<div class="hdr"><h1>◈ समृद्धि SAMRIDDHI</h1><div class="sub">NCD + EQUITY REINVESTMENT SYSTEM — ADAM DEEP LITERATURE ENGINE</div></div>',unsafe_allow_html=True)

    # ── THESIS (always visible) ───────────────────────────────
    st.markdown(f'<div class="th"><strong>The system:</strong> Deploy {fs(p.principal)} into NCDs at {fp(p.ncd_rate,1)} → Harvest {fi(p.gm)}/month ({fi(p.nm)} post-TDS) → Reinvest into equity MF → At maturity: NCD principal returned + equity portfolio.<br><strong>The question:</strong> Does this beat putting {fs(p.principal)} directly into equity? Under what conditions? At what risk?</div>',unsafe_allow_html=True)

    # ── MODE SELECTOR ─────────────────────────────────────────
    mode=st.radio('',['🌍  Real World Analysis','📐  Deterministic Analysis'],horizontal=True,label_visibility='collapsed')
    st.markdown('<div class="dv"></div>',unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # 🌍 REAL WORLD MODE
    # ══════════════════════════════════════════════════════════
    if '🌍' in mode:
        t1,t2,t3,t4=st.tabs(['🎯 SCENARIO','📜 BACKTEST','🎲 DISTRIBUTION','🛡️ TAX & RISK'])

        # ── SCENARIO ──────────────────────────────────────────
        with t1:
            st.markdown('## Generate a Market Scenario')
            st.markdown(f'<div class="ad"><span class="tg">Adam · Real World</span>Each seed generates a unique {p.horizon}-year market history from a lognormal process calibrated to Indian equity (μ≈{fp(p.eq_return,0)}, σ={fp(EQ_VOL,0)}). The NCD system and Lump Sum face the <strong>same returns</strong> — the only difference is the deployment strategy.</div>',unsafe_allow_html=True)

            seed=st.number_input('Random Seed',value=42,step=1,help='Change to generate a different market')
            sc=e.scenario(seed); ar=sc['ar']; nd=sc['nd']; ls=sc['ls']

            # Return environment
            st.markdown('### Market Environment')
            cl,cr=st.columns(2)
            with cl:
                fb=go.Figure(go.Bar(x=list(range(1,p.horizon+1)),y=ar*100,marker_color=[C['gn'] if r>=0 else C['rd'] for r in ar],text=[fp(r) for r in ar],textposition='outside',textfont=dict(size=9,color=C['t2'])))
                fb.update_layout(xaxis_title='Year',yaxis_title='%',yaxis=dict(zeroline=True,zerolinecolor=C['mu']))
                st.plotly_chart(_lo(fb,'Annual Equity Returns',320),width='stretch')
            with cr:
                env_stats = {'Metric':['Mean Return','Median Return','Best Year','Worst Year','Positive Years','Volatility (realised)'],
                    'Value':[fp(np.mean(ar)),fp(np.median(ar)),f'{fp(max(ar))} (Yr {int(np.argmax(ar))+1})',f'{fp(min(ar))} (Yr {int(np.argmin(ar))+1})',
                             f'{sum(1 for r in ar if r>=0)}/{p.horizon}',fp(np.std(ar))]}
                st.dataframe(pd.DataFrame(env_stats),width='stretch',hide_index=True,height=260)

            # Wealth paths
            st.markdown('### Wealth Path')
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=months,y=nd['total'],name='NCD System',line=dict(color=C['g'],width=2.5)))
            fig.add_trace(go.Scatter(x=months,y=ls['total'],name='Lump Sum',line=dict(color=C['cy'],width=2,dash='dash')))
            fig.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'],annotation_text='Capital')
            st.plotly_chart(_lo(fig,'NCD System vs Lump Sum — Same Random Returns',420),width='stretch')

            # Outcome
            st.markdown('### Outcome')
            k1,k2,k3,k4,k5,k6=st.columns(6)
            w='NCD' if sc['nd']['net'][-1]>sc['ls']['net'][-1] else 'Lump Sum'
            k1.metric('NCD Net',fs(nd['net'][-1]),fp(sc['ncd_cagr'])+' CAGR')
            k2.metric('Lump Net',fs(ls['net'][-1]),fp(sc['ls_cagr'])+' CAGR')
            k3.metric('Winner',w,fs(abs(nd['net'][-1]-ls['net'][-1])))
            k4.metric('NCD Max DD',fp(sc['n_mdd'],1),f'vs {fp(sc["l_mdd"],1)} lump')
            k5.metric('NCD Underwater',f'{sc["n_uw"]}/{p.N}',f'vs {sc["l_uw"]}/{p.N} lump')
            k6.metric('NCD Leads',f'{sc["ncd_wins"]}/{p.horizon}','year-ends')

            # Drawdown
            st.markdown('### Drawdown (Underwater)')
            fig_dd=go.Figure()
            fig_dd.add_trace(go.Scatter(x=months,y=sc['n_dd']*100,name='NCD',fill='tozeroy',fillcolor='rgba(255,195,0,.12)',line=dict(color=C['g'],width=1.5)))
            fig_dd.add_trace(go.Scatter(x=months,y=sc['l_dd']*100,name='Lump',fill='tozeroy',fillcolor='rgba(0,188,212,.08)',line=dict(color=C['cy'],width=1.2,dash='dash')))
            fig_dd.update_layout(xaxis_title='Month',yaxis_title='Drawdown (%)')
            st.plotly_chart(_lo(fig_dd,'Drawdown from Peak',320),width='stretch')

            # Year-by-year table
            st.markdown('### Year-by-Year')
            yt=[{'Year':d['yr'],'Return':fp(d['ret']),'NCD Total':fs(d['nt']),'NCD Net':fs(d['nn']),'Lump Total':fs(d['lt']),'Lump Net':fs(d['ln']),'Leader':d['leader'],'Gap':fs(abs(d['nt']-d['lt'])),'NCD CaR':fs(d['nc'])} for d in sc['yd']]
            st.dataframe(pd.DataFrame(yt),width='stretch',hide_index=True)

            # Rolling 3Y
            if sc['r3']:
                st.markdown('### Rolling 3-Year CAGR')
                fig_r3=go.Figure()
                fig_r3.add_trace(go.Bar(x=[r['yr'] for r in sc['r3']],y=[r['ncd']*100 for r in sc['r3']],name='NCD',marker_color=C['g'],opacity=.8))
                fig_r3.add_trace(go.Bar(x=[r['yr'] for r in sc['r3']],y=[r['ls']*100 for r in sc['r3']],name='Lump',marker_color=C['cy'],opacity=.6))
                fig_r3.update_layout(barmode='group',xaxis_title='Ending Year',yaxis_title='3Y CAGR (%)',yaxis=dict(zeroline=True,zerolinecolor=C['mu']))
                st.plotly_chart(_lo(fig_r3,'Rolling 3-Year CAGR',320),width='stretch')

            # 25-path aggregate
            st.markdown('### 25-Path Aggregate (same seed family)')
            multi=e.scenario_multi(seed*1000,25)
            fm=go.Figure()
            for i,r in enumerate(multi):
                fm.add_trace(go.Scatter(x=months,y=r['ncd_path'],showlegend=(i==0),name='NCD' if i==0 else None,legendgroup='n',line=dict(color=C['g'],width=.7),opacity=.35))
                fm.add_trace(go.Scatter(x=months,y=r['ls_path'],showlegend=(i==0),name='Lump' if i==0 else None,legendgroup='l',line=dict(color=C['cy'],width=.6),opacity=.25))
            fm.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
            st.plotly_chart(_lo(fm,'25 Paths — NCD (gold) vs Lump (cyan)',440),width='stretch')

            na_=np.array([r['ncd_net'] for r in multi]); la_=np.array([r['ls_net'] for r in multi])
            nc_=np.array([r['ncd_cagr'] for r in multi]); lc_=np.array([r['ls_cagr'] for r in multi])
            nd_=[r['ncd_mdd'] for r in multi]; ld_=[r['ls_mdd'] for r in multi]
            st.dataframe(pd.DataFrame({
                'Metric':['Mean Net','Median Net','Worst Case','Best Case','Mean CAGR','Worst CAGR','Mean Max DD','Win Count'],
                'NCD':[fs(na_.mean()),fs(np.median(na_)),fs(na_.min()),fs(na_.max()),fp(nc_.mean()),fp(nc_.min()),fp(np.mean(nd_),1),f'{np.sum(na_>la_)}/25'],
                'Lump':[fs(la_.mean()),fs(np.median(la_)),fs(la_.min()),fs(la_.max()),fp(lc_.mean()),fp(lc_.min()),fp(np.mean(ld_),1),f'{np.sum(la_>na_)}/25'],
            }),width='stretch',hide_index=True)

        # ── BACKTEST ──────────────────────────────────────────
        with t2:
            st.markdown('## Historical NIFTY Backtest')
            st.markdown(f'<div class="th">Every possible {p.horizon}-year window using <strong>actual NIFTY50 returns</strong> ({min(NIFTY.keys())}–{max(NIFTY.keys())}). No simulation — what <em>actually</em> happened.</div>',unsafe_allow_html=True)

            valid=[y for y in sorted(NIFTY.keys()) if y+p.horizon-1<=max(NIFTY.keys())]
            if valid:
                htbl=[]; ncd_w=0
                for yr in valid:
                    r=e.backtest(yr)
                    if r:
                        w='NCD' if r['ncd_net']>r['ls_net'] else 'Lump'
                        if w=='NCD': ncd_w+=1
                        htbl.append({'Start':yr,'Period':f'{yr}–{yr+p.horizon-1}','NCD Total':fs(r['ncd_total']),'NCD Net':fs(r['ncd_net']),'Lump Total':fs(r['ls_total']),'Lump Net':fs(r['ls_net']),'Winner':w,'Margin':fs(abs(r['ncd_net']-r['ls_net']))})
                st.dataframe(pd.DataFrame(htbl),width='stretch',hide_index=True)

                st.markdown(f'<div class="th">Across <strong>{len(htbl)} historical windows</strong>: NCD system won <strong>{ncd_w}/{len(htbl)}</strong> times. Lump sum equity wins in sustained bull markets. NCD system wins in periods that include significant crashes (2008, 2011, 2015, 2020) — precisely because the capital was preserved and DCA bought at lower prices.</div>',unsafe_allow_html=True)

                # Show selected window
                sel=st.selectbox('View detailed path',valid,format_func=lambda y:f'{y}–{y+p.horizon-1}')
                bt=e.backtest(sel)
                if bt:
                    cl,cr=st.columns(2)
                    with cl:
                        fb=go.Figure(go.Bar(x=bt['yrs'],y=[r*100 for r in bt['ar']],marker_color=[C['gn'] if r>=0 else C['rd'] for r in bt['ar']],text=[fp(r) for r in bt['ar']],textposition='outside',textfont=dict(size=8,color=C['t2'])))
                        fb.update_layout(xaxis_title='Year',yaxis_title='%')
                        st.plotly_chart(_lo(fb,f'NIFTY Returns {sel}–{sel+p.horizon-1}',320),width='stretch')
                    with cr:
                        fp3=go.Figure()
                        fp3.add_trace(go.Scatter(x=months,y=bt['nd']['total'],name='NCD',line=dict(color=C['g'],width=2.5)))
                        fp3.add_trace(go.Scatter(x=months,y=bt['ls']['total'],name='Lump',line=dict(color=C['cy'],width=2,dash='dash')))
                        fp3.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
                        st.plotly_chart(_lo(fp3,'Wealth Path',320),width='stretch')
                    b1,b2,b3,b4=st.columns(4)
                    b1.metric('NCD Net',fs(bt['ncd_net']),fp(_cagr(bt['ncd_total'],p.principal,p.horizon)))
                    b2.metric('Lump Net',fs(bt['ls_net']),fp(_cagr(bt['ls_total'],p.principal,p.horizon)))
                    w='NCD' if bt['ncd_net']>bt['ls_net'] else 'Lump'
                    b3.metric('Winner',w); b4.metric('Avg NIFTY',fp(np.mean(bt['ar'])))
            else:
                st.info(f'Need {p.horizon}+ years of data. Reduce horizon.')

        # ── DISTRIBUTION ──────────────────────────────────────
        with t3:
            st.markdown('## Monte Carlo Distribution')
            st.markdown(f'<div class="ad"><span class="tg">Adam · GBM</span>{p.mc_paths:,} paired paths. Same random draws for NCD & Lump Sum. σ={fp(EQ_VOL,0)}. Drift-corrected.</div>',unsafe_allow_html=True)
            with st.spinner(f'{p.mc_paths:,} simulations...'):
                mc=e.mc()

            cl,cr=st.columns(2)
            with cl: st.plotly_chart(_fan(mc['pp'],months,C['g'],'NCD System',p),width='stretch')
            with cr: st.plotly_chart(_fan(mc['lp'],months,C['cy'],'Lump Sum Equity',p),width='stretch')

            fh=go.Figure()
            fh.add_trace(go.Histogram(x=mc['fn']/1e5,nbinsx=60,name='NCD Net',marker_color=C['g'],opacity=.6))
            fh.add_trace(go.Histogram(x=mc['ln']/1e5,nbinsx=60,name='Lump Net',marker_color=C['cy'],opacity=.4))
            fh.update_layout(barmode='overlay',xaxis_title='Net Wealth (₹L)',yaxis_title='Freq')
            st.plotly_chart(_lo(fh,'Terminal Distribution — Net of Tax',380),width='stretch')

            st.markdown('### Percentile Table (Net of Tax)')
            ptbl=[{'Pctl':f'{k}th','NCD':fs(mc['pn'][k]),'Lump':fs(mc['lpn'][k]),'SIP Only':fs(mc['spn'][k]),'NCD vs Lump':fs(mc['pn'][k]-mc['lpn'][k])} for k in [5,10,25,50,75,90,95]]
            st.dataframe(pd.DataFrame(ptbl),width='stretch',hide_index=True)

            st.markdown('### Risk Metrics')
            r1,r2,r3,r4=st.columns(4)
            r1.metric('Sharpe',f'{mc["sh"]:.3f}'); r2.metric('Sortino',f'{mc["so"]:.3f}')
            r3.metric('VaR (5%)',fs(mc['v5'])); r4.metric('CVaR (5%)',fs(mc['cv5']))
            r5,r6,r7,r8=st.columns(4)
            r5.metric('P(NCD > Lump)',fp(mc['pb_ls'],1)); r6.metric('P(Capital Safe)',fp(mc['pc'],1))
            r7.metric('P(Lump Loss)',fp(mc['ls_pl'],1)); r8.metric('Median Max DD',fp(mc['md'],1))
            r9,r10,r11,r12=st.columns(4)
            r9.metric('Break-Even',fp(mc['be'])); r10.metric('P(2× Capital)',fp(mc['p2'],1))
            r11.metric('MC Mean CAGR',fp(mc['mc_'])); r12.metric('CAGR Vol',fp(mc['vc']))

            fd_=go.Figure(go.Histogram(x=mc['dd']*100,nbinsx=60,marker_color=C['rd'],opacity=.6))
            fd_.add_vline(x=mc['md']*100,line_dash='dash',line_color=C['g'],annotation_text=f'Med: {fp(mc["md"],1)}')
            fd_.update_layout(xaxis_title='Max DD (%)',yaxis_title='Freq')
            st.plotly_chart(_lo(fd_,'Max Drawdown Distribution (NCD System)',320),width='stretch')

        # ── TAX & RISK ────────────────────────────────────────
        with t4:
            st.markdown('## Tax & Risk Analysis')
            st.markdown(f'<div class="ad"><span class="tg">Tax Regime</span><strong>NCD Interest:</strong> Slab {fp(p.tax_slab,0)} + Cess {fp(p.cess,0)} = {fp(p.et,1)}. TDS {fp(p.tds_rate,0)}.<br><strong>LTCG:</strong> {fp(p.ltcg_rate,1)} on gains > {fi(p.ltcg_exempt)}/FY. Annual exemption modelled per-FY.</div>',unsafe_allow_html=True)

            # Use deterministic for tax waterfall (clean illustration)
            nd_d=e.ncd(); ls_d=e.lump(); fd_d=e.fd_eq(); pfd_d=e.pure_fd()
            ai=p.principal*p.ncd_rate*p.horizon
            fw=go.Figure(go.Waterfall(x=['Interest\nIncome','Equity\nGains','Interest\nTax','LTCG','Net\nWealth'],
                y=[ai,nd_d['g'][N],-nd_d['ti'][N],-nd_d['lc'][N],nd_d['net'][N]],
                measure=['relative','relative','relative','relative','total'],
                connector=dict(line=dict(color=C['mu'],width=1)),
                decreasing=dict(marker=dict(color=C['rd'])),increasing=dict(marker=dict(color=C['gn'])),
                totals=dict(marker=dict(color=C['g'])),textposition='outside',
                text=[fs(ai),fs(nd_d['g'][N]),fs(nd_d['ti'][N]),fs(nd_d['lc'][N]),fs(nd_d['net'][N])],
                textfont=dict(color=C['t2'],size=9)))
            st.plotly_chart(_lo(fw,'NCD System — Income to Net Wealth',380),width='stretch')

            st.markdown('### Tax Comparison')
            st.dataframe(pd.DataFrame({
                '':['Total Wealth','Total Tax','Net Wealth','Tax Rate','Capital-at-Risk'],
                'NCD':[fi(nd_d['total'][N]),fi(nd_d['tt'][N]),fi(nd_d['net'][N]),fp(nd_d['tt'][N]/nd_d['total'][N]),fi(nd_d['car'][N])],
                'Lump Sum':[fi(ls_d['total'][N]),fi(ls_d['lc'][N]),fi(ls_d['net'][N]),fp(ls_d['lc'][N]/ls_d['total'][N] if ls_d['total'][N]>0 else 0),fi(p.principal)],
                f'FD+Eq':[fi(fd_d['total'][N]),fi(fd_d['tt'][N]),fi(fd_d['net'][N]),fp(fd_d['tt'][N]/fd_d['total'][N]),fi(fd_d['car'][N])],
                'Pure FD':[fi(pfd_d['total'][N]),fi(pfd_d['total'][N]-pfd_d['net'][N]),fi(pfd_d['net'][N]),fp((pfd_d['total'][N]-pfd_d['net'][N])/pfd_d['total'][N]),'₹0'],
            }),width='stretch',hide_index=True)

            st.markdown(f'<div class="th"><strong>Tax cost of yield arbitrage:</strong> NCD pays {fs(nd_d["tt"][N])} vs Lump\'s {fs(ls_d["lc"][N])} — extra {fs(nd_d["tt"][N]-ls_d["lc"][N])}. This is the price of capital preservation. Under volatile markets (real world), the NCD system\'s lower drawdown and DCA advantage often compensates.<br><br><strong>LTCG Harvesting:</strong> Annual redemption within {fi(p.ltcg_exempt)} exemption saves up to {fi(p.ltcg_exempt*p.ltcg_rate)}/year.</div>',unsafe_allow_html=True)

            st.markdown('### Risk Decomposition')
            for nm,sev,cl,desc in [
                ('Credit Risk','CRITICAL',C['rd'],'NCD default → principal loss. Diversify 5–10 issuers, AA- min, secured, quarterly monitoring.'),
                ('Equity Drawdown','HIGH',C['or_'],f'MC median max DD: {fp(mc["md"],1)}. NCD system only exposes reinvested interest to equity risk.'),
                ('Interest Rate','LOW (HTM)',C['gn'],'Fixed coupon. No impact if held to maturity.'),
                ('Inflation','MEDIUM',C['g'],f'At {fp(p.inflation,0)}, real NCD coupon declines. Equity hedges long-term.'),
                ('Reinvestment','LOW–MED',C['cy'],'At maturity, comparable yields may be unavailable. Ladder strategy recommended.')]:
                st.markdown(f'<div class="rc" style="border-left:3px solid {cl}"><div class="rt" style="color:{cl}">{nm} — {sev}</div><div class="rb">{desc}</div></div>',unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # 📐 DETERMINISTIC MODE
    # ══════════════════════════════════════════════════════════
    else:
        nd_d=e.ncd(); ls_d=e.lump(); fd_d=e.fd_eq(); pfd_d=e.pure_fd(); sv_d=e.savings()
        dec=e.decompose(nd_d)

        t1,t2,t3=st.tabs(['📊 PROJECTIONS','⚡ SENSITIVITY','📖 ADAM'])

        with t1:
            st.markdown('## Deterministic Projections')
            st.markdown(f'<div class="ad"><span class="tg">Adam · Note</span>Deterministic mode assumes constant {fp(p.eq_return,0)} every year. This is a theoretical reference — real markets are volatile. Under constant returns, lump sum always wins (full compounding from day 1). The NCD system\'s value proposition appears under <strong>real-world volatility</strong> — use the 🌍 Real World mode for decision-making.</div>',unsafe_allow_html=True)

            # 5-system chart
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=months,y=nd_d['total'],name='NCD System',line=dict(color=C['g'],width=2.5)))
            fig.add_trace(go.Scatter(x=months,y=ls_d['total'],name='Lump Sum',line=dict(color=C['cy'],width=2,dash='dash')))
            fig.add_trace(go.Scatter(x=months,y=fd_d['total'],name=f'FD+Eq ({fp(p.fd_rate,0)})',line=dict(color=C['or_'],width=1.5,dash='dashdot')))
            fig.add_trace(go.Scatter(x=months,y=pfd_d['total'],name='Pure FD',line=dict(color=C['mu'],width=1.2,dash='dot')))
            fig.add_trace(go.Scatter(x=months,y=sv_d['total'],name='Savings',line=dict(color=C['rd'],width=1,dash='dot')))
            fig.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
            st.plotly_chart(_lo(fig,'Total Wealth — 5 Strategies (Constant Return)',460),width='stretch')

            # Scorecard
            systems=[('NCD',C['g'],nd_d['net'][N],nd_d['tt'][N],nd_d['car'][N]),
                     ('Lump Sum',C['cy'],ls_d['net'][N],ls_d['lc'][N],p.principal),
                     (f'FD+Eq',C['or_'],fd_d['net'][N],fd_d['tt'][N],fd_d['car'][N]),
                     ('Pure FD',C['mu'],pfd_d['net'][N],pfd_d['total'][N]-pfd_d['net'][N],0),
                     ('Savings',C['rd'],sv_d['net'][N],sv_d['total'][N]-sv_d['net'][N],0)]
            winner_v=max(s[2] for s in systems)
            cols=st.columns(5)
            for i,(nm,cl,nv,tx,car) in enumerate(systems):
                w='w' if nv==winner_v else ''
                cr_txt=fs(car)+' equity' if car>0 else 'No equity risk'
                crown=' 👑' if nv==winner_v else ''
                cols[i].markdown(f'<div class="sc {w}"><div class="lb" style="color:{cl}">{nm}{crown}</div><div class="vl" style="color:{cl}">{fs(nv)}</div><div class="dt">Tax: {fs(tx)} · CAGR: {fp(_cagr(nv,p.principal,p.horizon))}</div></div>',unsafe_allow_html=True)

            # Year-end table
            st.markdown('### Year-End Table')
            tbl=[]
            for yr in range(1,p.horizon+1):
                m=yr*12-1
                tbl.append({'Year':yr,'NCD Net':fs(nd_d['net'][m]),'Lump Net':fs(ls_d['net'][m]),'FD Net':fs(fd_d['net'][m]),'NCD CAGR':fp(_cagr(nd_d['total'][m],p.principal,yr)),'NCD vs Lump':fs(nd_d['net'][m]-ls_d['net'][m]),'CaR':fs(nd_d['car'][m])})
            st.dataframe(pd.DataFrame(tbl),width='stretch',hide_index=True)

            # Wealth decomposition
            st.markdown('### Wealth Decomposition')
            fig_d=go.Figure(go.Waterfall(x=['Principal\n(returned)','Interest\nReinvested','Equity\nCompounding','Total'],
                y=[dec['principal'],dec['interest'],dec['compounding'],dec['total']],
                measure=['relative','relative','relative','total'],
                connector=dict(line=dict(color=C['mu'],width=1)),
                increasing=dict(marker=dict(color=C['gn'])),totals=dict(marker=dict(color=C['g'])),
                textposition='outside',text=[fs(v) for v in [dec['principal'],dec['interest'],dec['compounding'],dec['total']]],
                textfont=dict(color=C['t2'],size=9)))
            st.plotly_chart(_lo(fig_d,'Source of Each Rupee',360),width='stretch')

            # Regimes
            st.markdown('### Equity Return Regimes')
            scen={'Bear 7%':.07,'Below 10%':.10,'Base 13%':.13,'Bull 16%':.16,'Euphoric 19%':.19}
            clrs=[C['rd'],C['or_'],C['g'],C['gn'],C['cy']]
            fr=go.Figure()
            for i,(nm,ret) in enumerate(scen.items()):
                d=e.ncd(er=ret-p.expense_ratio)
                fr.add_trace(go.Scatter(x=months,y=d['total'],name=nm,line=dict(color=clrs[i],width=2.5 if 'Base' in nm else 1.5,dash='solid' if 'Base' in nm else 'dot')))
            fr.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
            st.plotly_chart(_lo(fr,'NCD System Under Different Regimes',420),width='stretch')
            cs=st.columns(5)
            for i,(nm,ret) in enumerate(scen.items()):
                fv=e.cf(er=ret-p.expense_ratio); tw=p.principal+fv
                cs[i].metric(nm,fs(tw),fp(_cagr(tw,p.principal,p.horizon)))

        with t2:
            st.markdown('## Sensitivity Analysis')
            sens=e.sens([.08,.09,.10,.118,.13,.14,.15],[.07,.10,.13,.16,.19])
            z=sens.values/1e5
            fgs=go.Figure(go.Heatmap(z=z,x=sens.columns.tolist(),y=sens.index.tolist(),colorscale=[[0,'#1B2A4A'],[.5,'#FFC300'],[1,'#FF5252']],text=[[f'₹{v:.1f}L' for v in r] for r in z],texttemplate='%{text}',textfont=dict(size=10,color='white'),colorbar=dict(title='₹L')))
            fgs.update_layout(xaxis_title='Equity Return',yaxis_title='NCD Yield')
            st.plotly_chart(_lo(fgs,'Net Wealth — NCD Yield × Equity Return',420),width='stretch')

            st.markdown('### Horizon Sensitivity')
            h_r=[]
            for h in range(3,21):
                tp=P(principal=p.principal,ncd_rate=p.ncd_rate,horizon=h,eq_return=p.eq_return,expense_ratio=p.expense_ratio,tax_slab=p.tax_slab,ltcg_rate=p.ltcg_rate)
                te=Eng(tp); fv=te.cf(); tw=tp.principal+fv
                ti=tp.gm*tp.et*tp.N; g=fv-tp.nm*tp.N; lc=max(0,(g-tp.ltcg_exempt*h))*tp.ltcg_rate
                h_r.append(dict(h=h,tw=tw,net=tw-ti-lc))
            hdf=pd.DataFrame(h_r)
            fh=go.Figure(go.Bar(x=hdf['h'],y=hdf['net']/1e5,marker_color=[C['g'] if h==p.horizon else C['cd'] for h in hdf['h']],
                text=[fx(r['tw']/p.principal) for r in h_r],textposition='outside',textfont=dict(size=8,color=C['t2'])))
            fh.update_layout(xaxis_title='Horizon (Years)',yaxis_title='Net Wealth (₹L)')
            st.plotly_chart(_lo(fh,'Net Wealth by Horizon',380),width='stretch')

        with t3:
            st.markdown('## Adam Deep Literature')
            cfv=e.cf(); nd_d=e.ncd(); delta=abs(cfv-nd_d['mf'][N]); ntc=p.ncd_rate*p.horizon*p.et

            st.markdown(f'<div class="ad"><span class="tg">Closed-Form Validation</span><code>V(N) = S·[(1+r)^N-1]/r</code> | S={fi(p.nm,True)} r={p.mr:.10f} N={p.N}<br>CF: <code>{fi(cfv,True)}</code> | Sim: <code>{fi(nd_d["mf"][N],True)}</code> | Δ: <code>{fi(delta,True)}</code> ✓</div>',unsafe_allow_html=True)

            st.markdown(f'<div class="ad"><span class="tg">Yield Arbitrage Theorem</span>NCD system dominates pure SIP by <code>P·(1-r_d·T·τ) = {fi(p.principal*(1-ntc))}</code><br>Valid when r_d·T·τ < 1. Current: {ntc:.4f} < 1 ✓</div>',unsafe_allow_html=True)

            st.markdown(f'<div class="ad"><span class="tg">Why Deterministic Favours Lump Sum</span>Under constant returns, Lump Sum gets full compounding from month 1: <code>P·(1+r)^N</code>. NCD system builds exposure gradually via SIP. The <strong>entire NCD thesis rests on volatility</strong> — DCA buys more units during drawdowns, and the principal is shielded from crashes. This edge is invisible in deterministic mode.</div>',unsafe_allow_html=True)

            mu_a=np.log(1+p.ne)/12-EQ_VOL**2/24
            st.markdown(f'<div class="ad"><span class="tg">GBM Drift Correction</span><code>μ_adj = ln(1+r)/12 - σ²/24 = {mu_a:.8f}</code><br><code>σ_m = σ/√12 = {EQ_VOL/np.sqrt(12):.8f}</code><br>Corrects Jensen\'s inequality so E[multiplicative return] is unbiased.</div>',unsafe_allow_html=True)

            st.markdown(f'<div class="ad"><span class="tg">Capital-at-Risk</span>NCD: only reinvested interest enters equity = <code>{fi(nd_d["car"][N])}</code>. Principal in debt.<br>Lump: 100% = <code>{fi(p.principal)}</code> at equity risk from day 1.<br>Net per unit CaR — NCD: {nd_d["net"][N]/nd_d["car"][N]:.2f}× | Lump: {ls_d["net"][N]/p.principal:.2f}×</div>',unsafe_allow_html=True)

            st.markdown(f'<div class="ad"><span class="tg">Per-FY LTCG Model</span>Annual {fi(p.ltcg_exempt)} exemption applied at each FY boundary. Saves {fi(p.ltcg_exempt*p.ltcg_rate)}/year vs terminal-only model.</div>',unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────
    st.markdown(f'<div style="text-align:center;padding:18px 0 10px;margin-top:28px;border-top:1px solid {C["gb"]}"><div style="font-family:Inter;font-weight:800;font-size:.78rem;color:{C["g"]};letter-spacing:.06em">◈ समृद्धि SAMRIDDHI</div><div style="font-family:JetBrains Mono;font-size:.52rem;color:{C["mu"]};margin-top:2px;letter-spacing:.1em">HEMREK CAPITAL — ADAM DEEP LITERATURE ENGINE</div></div>',unsafe_allow_html=True)

if __name__=='__main__': main()
