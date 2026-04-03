"""
◈ वृद्धि VRIDDHI — NCD + Equity Reinvestment Intelligence System
Hemrek Capital | Adam Deep Literature Engine

वृद्धि (Vriddhi): In classical Sanskrit financial texts, vriddhi denotes
the interest or yield that capital generates — the precise term for a
system that converts debt yield into equity wealth.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

EQUITY_VOL = 0.20

def _ic(n):
    s = str(abs(int(n)))
    if len(s) <= 3: return s
    r = s[-3:]
    s = s[:-3]
    while s:
        r = s[-2:] + ',' + r
        s = s[:-2]
    return r

def fi(v, paise=False):
    sgn = '-' if v < 0 else ''
    a = abs(v)
    if paise: return f'{sgn}₹{_ic(int(a))}.{round((a-int(a))*100):02d}'
    return f'{sgn}₹{_ic(round(a))}'

def fl(v, d=2): return f'₹{v/1e5:.{d}f}L'
def fc(v, d=2): return f'₹{v/1e7:.{d}f}Cr'

def fs(v):
    a = abs(v)
    if a >= 1e7: return fc(v)
    if a >= 1e5: return fl(v)
    return fi(v)

def fp(v, d=2): return f'{v*100:.{d}f}%'
def fx(v): return f'{v:.2f}x'

C = dict(bg='#0A0E17', bg2='#111827', card='#1A1F2E', inp='#0D1117',
         gold='#FFC300', gold_dim='#B8960F', gold_bg='rgba(255,195,0,0.08)',
         gold_b='rgba(255,195,0,0.25)', tx='#E8EAED', tx2='#9AA0A6', mu='#5F6368',
         grn='#00E676', red='#FF5252', blu='#448AFF', cyn='#00BCD4',
         pur='#B388FF', org='#FF9100', bdr='rgba(255,255,255,0.06)')

def inject_css():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp{{background:{C['bg']};color:{C['tx']};font-family:'Inter',sans-serif}}
    .stApp>header{{background:transparent!important}}
    section[data-testid="stSidebar"]{{background:{C['bg2']}!important;border-right:1px solid {C['gold_b']}!important}}
    section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] label{{color:{C['tx2']}!important;font-size:.85rem!important}}
    h1,h2,h3{{font-family:'Inter',sans-serif!important;letter-spacing:-.02em}}
    h1{{font-weight:800!important;font-size:1.6rem!important;color:{C['tx']}!important}}
    h2{{font-weight:700!important;font-size:1.2rem!important;color:{C['gold']}!important}}
    h3{{font-weight:600!important;font-size:1rem!important;color:{C['tx']}!important}}
    [data-testid="stMetric"]{{background:{C['card']};border:1px solid {C['gold_b']};border-radius:8px;padding:14px 16px}}
    [data-testid="stMetric"]:hover{{border-color:{C['gold']};box-shadow:0 0 15px rgba(255,195,0,.08)}}
    [data-testid="stMetricLabel"] p{{color:{C['mu']}!important;font-family:'Inter'!important;font-size:.68rem!important;font-weight:600!important;text-transform:uppercase;letter-spacing:.08em}}
    [data-testid="stMetricValue"]{{color:{C['gold']}!important;font-family:'JetBrains Mono',monospace!important;font-weight:600!important;font-size:1.2rem!important}}
    [data-testid="stMetricDelta"]{{font-family:'JetBrains Mono',monospace!important;font-size:.73rem!important}}
    .stNumberInput input,.stTextInput input{{background:{C['inp']}!important;color:{C['gold']}!important;border:1px solid {C['gold_b']}!important;border-radius:6px!important;font-family:'JetBrains Mono',monospace!important}}
    .stTabs [data-baseweb="tab-list"]{{gap:2px;background:{C['bg2']};border-radius:8px;padding:3px}}
    .stTabs [data-baseweb="tab"]{{background:transparent;color:{C['mu']};border-radius:6px;font-family:'Inter';font-weight:600;font-size:.78rem}}
    .stTabs [aria-selected="true"]{{background:{C['card']}!important;color:{C['gold']}!important;border-bottom:2px solid {C['gold']}!important}}
    .streamlit-expanderHeader{{background:{C['card']}!important;border:1px solid {C['bdr']}!important;border-radius:6px!important;color:{C['tx2']}!important;font-family:'JetBrains Mono',monospace!important;font-size:.83rem!important}}
    .streamlit-expanderContent{{background:{C['card']}!important;border:1px solid {C['bdr']}!important;border-top:none!important}}
    .hdr{{background:linear-gradient(135deg,{C['bg2']},{C['card']});border:1px solid {C['gold_b']};border-left:3px solid {C['gold']};border-radius:8px;padding:16px 20px;margin-bottom:18px}}
    .hdr h1{{margin:0!important;padding:0!important;font-size:1.4rem!important}}
    .hdr .sub{{color:{C['mu']};font-family:'JetBrains Mono',monospace;font-size:.7rem;margin-top:4px;letter-spacing:.05em}}
    .adam{{background:{C['card']};border:1px solid {C['bdr']};border-left:3px solid {C['pur']};border-radius:6px;padding:13px 17px;margin:10px 0;font-family:'JetBrains Mono',monospace;font-size:.79rem;color:{C['tx2']};line-height:1.6}}
    .adam .tag{{color:{C['pur']};font-weight:700;font-size:.66rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:5px;display:block}}
    .adam code{{color:{C['gold']};background:rgba(255,195,0,.08);padding:1px 5px;border-radius:3px}}
    .th{{background:linear-gradient(135deg,rgba(255,195,0,.04),rgba(255,195,0,.01));border:1px solid {C['gold_b']};border-radius:8px;padding:16px 20px;margin:10px 0;color:{C['tx2']};font-size:.84rem;line-height:1.65}}
    .th strong{{color:{C['gold']}}}
    .dv{{height:1px;background:linear-gradient(to right,transparent,{C['gold_b']},transparent);margin:22px 0}}
    .sc{{text-align:center;padding:12px;background:{C['card']};border-radius:8px;border:1px solid {C['bdr']}}}
    .sc.act{{border-color:{C['gold_b']}}}
    .sc .lb{{font-size:.63rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700}}
    .sc .vl{{font-family:'JetBrains Mono',monospace;font-size:1.25rem;font-weight:700;margin:5px 0}}
    .sc .dt{{font-size:.73rem;color:{C['tx2']}}}
    .rc{{background:{C['card']};border:1px solid {C['bdr']};border-radius:8px;padding:13px 17px;margin:5px 0}}
    .rc .rt{{font-weight:700;font-size:.86rem;margin-bottom:2px}}
    .rc .rb{{color:{C['tx2']};font-size:.79rem;line-height:1.5}}
    </style>""", unsafe_allow_html=True)

@dataclass
class Pm:
    principal:float=10_00_000; ncd_rate:float=0.118; horizon:int=10
    eq_return:float=0.13; expense_ratio:float=0.005
    tax_slab:float=0.30; cess:float=0.04; tds_rate:float=0.10
    ltcg_rate:float=0.125; stcg_rate:float=0.20; ltcg_exempt:float=1_25_000
    surcharge:float=0.0; inflation:float=0.06; fd_rate:float=0.07
    savings_rate:float=0.04; mc_paths:int=5000
    @property
    def eff_tax(self): return self.tax_slab*(1+self.cess)*(1+self.surcharge)
    @property
    def gross_m(self): return self.principal*self.ncd_rate/12
    @property
    def net_m(self): return self.gross_m*(1-self.tds_rate)
    @property
    def net_eq(self): return self.eq_return-self.expense_ratio
    @property
    def mr(self): return (1+self.net_eq)**(1/12)-1
    @property
    def N(self): return self.horizon*12

class Eng:
    def __init__(self, p): self.p = p

    def _sim(self, sip, rets, princ=0):
        p=self.p; rows=[]; mf=cum=0.0; scalar=np.isscalar(rets)
        for m in range(p.N):
            r = rets if scalar else rets[m]
            mf = mf*(1+r)+sip; cum+=sip; g=mf-cum
            ltcg=max(0,(g-p.ltcg_exempt))*p.ltcg_rate
            rows.append(dict(month=m+1,year=m//12+1,mf=mf,cum_inv=cum,gains=g,ltcg=ltcg,
                             total=princ+mf,net=princ+mf-ltcg,r_m=r))
        return pd.DataFrame(rows)

    def ncd(self, er=None):
        p=self.p; r=(1+(er if er is not None else p.net_eq))**(1/12)-1
        b=self._sim(p.net_m, r, p.principal)
        b['cum_tax_int']=p.gross_m*p.eff_tax*b['month']
        b['total_tax']=b['cum_tax_int']+b['ltcg']
        b['net_wealth']=b['total']-b['total_tax']
        return b

    def sip(self, er=None):
        p=self.p; r=(1+(er if er is not None else p.net_eq))**(1/12)-1
        return self._sim(p.net_m, r, 0)

    def fd_sys(self):
        p=self.p; fd_nm=p.principal*p.fd_rate/12*(1-p.tds_rate)
        b=self._sim(fd_nm, p.mr, p.principal)
        b['cum_tax_int']=p.principal*p.fd_rate/12*p.eff_tax*b['month']
        b['total_tax']=b['cum_tax_int']+b['ltcg']
        b['net_wealth']=b['total']-b['total_tax']
        return b

    def cf(self, sip=None, er=None):
        p=self.p; S=sip or p.net_m; r=(1+(er if er is not None else p.net_eq))**(1/12)-1
        return S*((1+r)**p.N-1)/r if r!=0 else S*p.N

    def stress(self, seed, n=1):
        p=self.p; mu=np.log(1+p.net_eq)/12-EQUITY_VOL**2/24; sig=EQUITY_VOL/np.sqrt(12)
        np.random.seed(seed); res=[]
        for _ in range(n):
            lr=np.random.normal(mu,sig,p.N); mr=np.exp(lr)-1
            nm=sm=cum=0.0; nv=[]; sv=[]
            for m in range(p.N):
                nm=nm*(1+mr[m])+p.net_m; sm=sm*(1+mr[m])+p.net_m; cum+=p.net_m
                nv.append(p.principal+nm); sv.append(sm)
            ar=[np.prod(1+mr[y*12:(y+1)*12])-1 for y in range(p.horizon)]
            ng=nm-cum; nl=max(0,(ng-p.ltcg_exempt))*p.ltcg_rate
            nn=p.principal+nm-p.gross_m*p.eff_tax*p.N-nl
            sg=sm-cum; sl=max(0,(sg-p.ltcg_exempt))*p.ltcg_rate; sn=sm-sl
            res.append(dict(mr=mr,ar=np.array(ar),np_=np.array(nv),sp=np.array(sv),
                            nt=p.principal+nm,nn=nn,nm_=nm,ng=ng,st=sm,sn=sn,cum=cum))
        return res

    def mc(self):
        p=self.p; mu=np.log(1+p.net_eq)/12-EQUITY_VOL**2/24; sig=EQUITY_VOL/np.sqrt(12)
        np.random.seed(42); lr=np.random.normal(mu,sig,(p.mc_paths,p.N)); mr=np.exp(lr)-1
        mf=np.zeros((p.mc_paths,p.N)); smf=np.zeros((p.mc_paths,p.N))
        for m in range(p.N):
            prev=mf[:,m-1] if m>0 else 0.0; mf[:,m]=prev*(1+mr[:,m])+p.net_m
            prev2=smf[:,m-1] if m>0 else 0.0; smf[:,m]=prev2*(1+mr[:,m])+p.net_m
        ci=p.net_m*np.arange(1,p.N+1); tot=p.principal+mf; ft=tot[:,-1]; fm=mf[:,-1]
        fg=fm-ci[-1]; it=p.gross_m*p.eff_tax*p.N; lc=np.maximum(0,(fg-p.ltcg_exempt))*p.ltcg_rate
        fn=ft-it-lc; sf=smf[:,-1]; sg=sf-ci[-1]; sl=np.maximum(0,(sg-p.ltcg_exempt))*p.ltcg_rate; sn=sf-sl
        pk=[5,10,25,50,75,90,95]
        pp={k:np.percentile(tot,k,axis=0) for k in pk}; sp={k:np.percentile(smf,k,axis=0) for k in pk}
        dd=np.zeros(p.mc_paths)
        for i in range(p.mc_paths):
            pk_=np.maximum.accumulate(tot[i]); dd[i]=((tot[i]-pk_)/pk_).min()
        return dict(ft=ft,fn=fn,sf=sf,sn=sn,pp=pp,sp=sp,dd=dd,
                    pct={k:np.percentile(ft,k) for k in pk},
                    pn={k:np.percentile(fn,k) for k in pk},
                    spn={k:np.percentile(sn,k) for k in pk})

    def met(self, nd, si, fd, mc):
        p=self.p; nf=nd.iloc[-1]; sf=si.iloc[-1]; ff=fd.iloc[-1]
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
                fv=p.net_m*((1+rm)**p.N-1)/rm if rm!=0 else p.net_m*p.N
                g=fv-p.net_m*p.N; return fv-p.gross_m*p.eff_tax*p.N-max(0,(g-p.ltcg_exempt))*p.ltcg_rate
            be=brentq(_w,-.10,.40)
        except: be=0.0
        sv=p.principal*(1+p.savings_rate)**p.horizon; st_=(sv-p.principal)*p.eff_tax
        return dict(
            nt=nf['total'],nn=nf['net_wealth'],nm=nf['mf'],ng=nf['gains'],
            nci=nf['cum_inv'],nti=nf['cum_tax_int'],nl=nf['ltcg'],ntt=nf['total_tax'],
            nc=(nf['total']/p.principal)**(1/p.horizon)-1,
            ncp=(nf['net_wealth']/p.principal)**(1/p.horizon)-1,
            nmx=nf['total']/p.principal,
            st_=sf['mf'],sn=sf['net'],sg=sf['gains'],sl=sf['ltcg'],sci=sf['cum_inv'],
            ft_=ff['total'],fn_=ff['net_wealth'],svn=sv-st_,
            adv=nf['net_wealth']-sf['net'],advf=nf['net_wealth']-ff['net_wealth'],
            sh=sh,so=so,om=om,ca=ca,v5=v5,cv5=cv5,mc_=mc_,vc=vc,
            pb=np.mean(mc['ft']>mc['sf']),pc=np.mean(mc['fn']>p.principal),
            p2=np.mean(mc['ft']>=2*p.principal),pl=np.mean(mc['fn']<p.principal),
            md=md,be=be)

    def sens(self, nys, ers):
        p=self.p; rows=[]
        for ny in nys:
            row={}
            for er in ers:
                s=p.principal*ny/12*(1-p.tds_rate); rm=(1+er-p.expense_ratio)**(1/12)-1
                fv=s*((1+rm)**p.N-1)/rm if rm!=0 else s*p.N; v=p.principal+fv
                v-=p.principal*ny*p.horizon*p.eff_tax
                g=fv-s*p.N; v-=max(0,(g-p.ltcg_exempt))*p.ltcg_rate
                row[f'{er:.0%}']=v
            rows.append(row)
        return pd.DataFrame(rows, index=[f'{y:.1%}' for y in nys])


def _lo(fig, title='', h=480):
    fig.update_layout(
        title=dict(text=title, font=dict(color=C['gold'], size=13, family='Inter')),
        paper_bgcolor=C['card'], plot_bgcolor=C['card'],
        font=dict(color=C['tx2'], family='JetBrains Mono, monospace', size=10),
        xaxis=dict(gridcolor='rgba(255,255,255,.04)', zerolinecolor='rgba(255,255,255,.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,.04)', zerolinecolor='rgba(255,255,255,.06)'),
        height=h, margin=dict(t=50,b=40,l=60,r=20),
        legend=dict(bgcolor='rgba(0,0,0,.3)', bordercolor=C['gold_b'], borderwidth=1, font=dict(size=9)),
        hoverlabel=dict(bgcolor=C['bg2'], font_size=10, font_family='JetBrains Mono'))
    return fig


def main():
    st.set_page_config(page_title='Vriddhi — Hemrek Capital', page_icon='◈', layout='wide', initial_sidebar_state='expanded')
    inject_css()

    with st.sidebar:
        st.markdown(f"""<div style="text-align:center;padding:8px 0 12px"><div style="font-family:'Inter';font-weight:800;font-size:1.05rem;color:{C['gold']};letter-spacing:.04em">◈ वृद्धि VRIDDHI</div><div style="font-family:'JetBrains Mono';font-size:.6rem;color:{C['mu']};letter-spacing:.12em;margin-top:2px">HEMREK CAPITAL</div></div><div style="height:1px;background:{C['gold_b']};margin:0 0 12px"></div>""", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{C['gold']};font-size:.7rem;font-weight:700;letter-spacing:.1em'>NCD PARAMETERS</p>", unsafe_allow_html=True)
        principal=st.number_input('Investment (₹)',value=10_00_000,step=1_00_000,format='%d')
        ncd_rate=st.slider('NCD Coupon (%)',6.0,16.0,11.8,0.1)/100
        horizon=st.slider('Horizon (Years)',3,20,10)
        st.markdown(f"<p style='color:{C['gold']};font-size:.7rem;font-weight:700;letter-spacing:.1em;margin-top:12px'>EQUITY</p>", unsafe_allow_html=True)
        eq_return=st.slider('Expected Return (%)',5.0,22.0,13.0,0.5)/100
        expense_ratio=st.slider('Expense Ratio (%)',0.0,2.5,0.5,0.1)/100
        st.markdown(f"<p style='color:{C['gold']};font-size:.7rem;font-weight:700;letter-spacing:.1em;margin-top:12px'>TAX (INDIA)</p>", unsafe_allow_html=True)
        tax_slab=st.selectbox('Tax Slab',[0.0,.05,.10,.15,.20,.25,.30],index=6,format_func=lambda x:f'{x:.0%}')
        ltcg_rate=st.number_input('LTCG Rate (%)',value=12.5,step=0.5)/100
        fd_rate=st.slider('FD Rate (%)',4.0,9.0,7.0,0.25)/100
        st.markdown(f"<p style='color:{C['gold']};font-size:.7rem;font-weight:700;letter-spacing:.1em;margin-top:12px'>ENVIRONMENT</p>", unsafe_allow_html=True)
        inflation=st.slider('Inflation (%)',2.0,10.0,6.0,0.5)/100
        mc_paths=st.select_slider('MC Paths',[1000,2500,5000,10000],value=5000)

    p=Pm(principal=principal,ncd_rate=ncd_rate,horizon=horizon,eq_return=eq_return,
         expense_ratio=expense_ratio,tax_slab=tax_slab,ltcg_rate=ltcg_rate,
         fd_rate=fd_rate,inflation=inflation,mc_paths=mc_paths)
    e=Eng(p); nd=e.ncd(); si=e.sip(); fd=e.fd_sys()
    nf=nd.iloc[-1]; sf=si.iloc[-1]; ff=fd.iloc[-1]

    st.markdown(f'<div class="hdr"><h1>◈ वृद्धि VRIDDHI</h1><div class="sub">NCD + EQUITY REINVESTMENT SYSTEM — ADAM DEEP LITERATURE ENGINE</div></div>', unsafe_allow_html=True)

    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric('Total Wealth',fs(nf['total']),fx(nf['total']/p.principal))
    k2.metric('Net of All Tax',fs(nf['net_wealth']),fp((nf['net_wealth']/p.principal)**(1/p.horizon)-1)+' CAGR')
    k3.metric('MF Portfolio',fs(nf['mf']),f'{fs(nf["gains"])} gains')
    k4.metric('Monthly SIP',fi(p.net_m),f'from {fp(p.ncd_rate,1)} yield')
    k5.metric('NCD Principal',fs(p.principal),'Returned at maturity')
    k6.metric('vs Pure SIP',fs(nf['net_wealth']-sf['net']),'net advantage')
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6=st.tabs(['◈ OVERVIEW','📊 PROJECTIONS','🎯 STRESS TEST','🎲 MONTE CARLO','🏛️ TAX & COMPARISON','📖 ADAM'])

    with t1:
        st.markdown('## Investment Thesis')
        st.markdown(f'<div class="th"><strong>Deploy</strong> {fs(p.principal)} into NCDs at {fp(p.ncd_rate,1)} → <strong>Harvest</strong> {fi(p.gross_m)}/month ({fi(p.net_m)} post-TDS) → <strong>Reinvest</strong> into equity MF at {fp(p.eq_return,0)} → <strong>At maturity:</strong> NCD principal returned + equity portfolio of {fs(nf["mf"])} = <strong>{fs(nf["total"])}</strong> ({fx(nf["total"]/p.principal)}, {fp((nf["total"]/p.principal)**(1/p.horizon)-1)} CAGR)</div>', unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=nd['month'],y=nd['total'],name='NCD System',line=dict(color=C['gold'],width=2.5)))
        fig.add_trace(go.Scatter(x=nd['month'],y=nd['mf'],name='MF Component',line=dict(color=C['cyn'],width=1.3,dash='dot')))
        fig.add_trace(go.Scatter(x=si['month'],y=si['mf'],name='Pure SIP',line=dict(color=C['red'],width=1.8,dash='dash')))
        fig.add_trace(go.Scatter(x=fd['month'],y=fd['total'],name=f'FD+Eq ({fp(p.fd_rate,0)})',line=dict(color=C['org'],width=1.3,dash='dashdot')))
        fig.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'],annotation_text='Capital Floor')
        st.plotly_chart(_lo(fig,'Wealth Accumulation — All Systems',480),use_container_width=True)

        st.markdown('## Comparative Scorecard')
        sv=p.principal*(1+p.savings_rate)**p.horizon; svt=(sv-p.principal)*p.eff_tax
        cards=[('NCD System',C['gold'],fs(nf['net_wealth']),f'CAGR {fp((nf["net_wealth"]/p.principal)**(1/p.horizon)-1)} · Tax {fs(nf["total_tax"])}',True),
               ('Pure SIP',C['red'],fs(sf['net']),f'Tax {fs(sf["ltcg"])} · No principal',False),
               (f'FD+Eq ({fp(p.fd_rate,0)})',C['org'],fs(ff['net_wealth']),'Lower yield → smaller SIP',False),
               (f'Savings ({fp(p.savings_rate,0)})',C['mu'],fs(sv-svt),'Idle capital benchmark',False)]
        cols=st.columns(4)
        for i,(nm,cl,vl,dt,act) in enumerate(cards):
            cls='sc act' if act else 'sc'
            cols[i].markdown(f'<div class="{cls}"><div class="lb" style="color:{cl}">{nm}</div><div class="vl" style="color:{cl}">{vl}</div><div class="dt">{dt}</div></div>',unsafe_allow_html=True)

        ny_=nd[nd['month']%12==0]; sy_=si[si['month']%12==0]; fy_=fd[fd['month']%12==0]
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=ny_['year'],y=ny_['net_wealth'],name='NCD Net',mode='lines+markers',line=dict(color=C['gold'],width=2.5),marker=dict(size=6)))
        fig2.add_trace(go.Scatter(x=sy_['year'],y=sy_['net'],name='SIP Net',mode='lines+markers',line=dict(color=C['red'],width=2,dash='dash'),marker=dict(size=5)))
        fig2.add_trace(go.Scatter(x=fy_['year'],y=fy_['net_wealth'],name='FD Net',mode='lines+markers',line=dict(color=C['org'],width=1.5,dash='dashdot'),marker=dict(size=4)))
        fig2.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
        fig2.update_layout(yaxis_title='Net-of-Tax (₹)')
        st.plotly_chart(_lo(fig2,'Post-Tax Wealth — Year by Year',420),use_container_width=True)

    with t2:
        st.markdown('## Deterministic Projections')
        cfv=e.cf(); delta=abs(cfv-nf['mf'])
        st.markdown(f'<div class="adam"><span class="tag">Adam · Closed-Form</span><code>V(N) = S·[(1+r)^N-1]/r</code> = <code>{fi(cfv,True)}</code> | Sim: <code>{fi(nf["mf"],True)}</code> | Δ: <code>{fi(delta,True)}</code> ✓</div>',unsafe_allow_html=True)
        st.markdown('### Year-End Comparison')
        tbl=[]
        for _,r in ny_.iterrows():
            yr=int(r['year']); s=sy_[sy_['year']==yr].iloc[0]; f=fy_[fy_['year']==yr].iloc[0]
            tbl.append({'Year':yr,'NCD Total':fs(r['total']),'NCD Net':fs(r['net_wealth']),'SIP Value':fs(s['mf']),'SIP Net':fs(s['net']),'FD Net':fs(f['net_wealth']),'Adv vs SIP':fs(r['net_wealth']-s['net']),'NCD CAGR':fp((r['total']/p.principal)**(1/yr)-1)})
        st.dataframe(pd.DataFrame(tbl),use_container_width=True,hide_index=True)

        st.markdown('### Regime Scenarios')
        scen={'Bear 7%':.07,'Below 10%':.10,'Base 13%':.13,'Bull 16%':.16,'Euphoric 19%':.19}
        clrs=[C['red'],C['org'],C['gold'],C['grn'],C['cyn']]
        fr=go.Figure()
        for i,(nm,ret) in enumerate(scen.items()):
            d=e.ncd(er=ret-p.expense_ratio)
            fr.add_trace(go.Scatter(x=d['month'],y=d['total'],name=nm,line=dict(color=clrs[i],width=2.5 if 'Base' in nm else 1.5,dash='solid' if 'Base' in nm else 'dot')))
        fr.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
        st.plotly_chart(_lo(fr,'Total Wealth Across Regimes',460),use_container_width=True)
        cs=st.columns(5)
        for i,(nm,ret) in enumerate(scen.items()):
            fv=e.cf(er=ret-p.expense_ratio); tw=p.principal+fv
            cs[i].metric(nm,fs(tw),fp((tw/p.principal)**(1/p.horizon)-1))

        st.markdown('### Sensitivity Matrix (Net Wealth)')
        sens=e.sens([.08,.09,.10,.118,.13,.14,.15],[.07,.10,.13,.16,.19])
        z=sens.values/1e5
        fgs=go.Figure(go.Heatmap(z=z,x=sens.columns.tolist(),y=sens.index.tolist(),colorscale=[[0,'#1B2A4A'],[.5,'#FFC300'],[1,'#FF5252']],text=[[f'₹{v:.1f}L' for v in row] for row in z],texttemplate='%{text}',textfont=dict(size=10,color='white'),colorbar=dict(title='₹L')))
        fgs.update_layout(xaxis_title='Equity Return',yaxis_title='NCD Yield')
        st.plotly_chart(_lo(fgs,'NCD Yield × Equity Return → Net Wealth',440),use_container_width=True)

    with t3:
        st.markdown('## Random Returns Stress Test')
        st.markdown(f'<div class="th">Unlike the deterministic projection (constant {fp(p.eq_return,0)}/year), this generates <strong>random year-by-year equity returns</strong> calibrated to Indian equity markets (≈{fp(EQUITY_VOL,0)} annual volatility). Each seed produces a unique sequence — some years surge, others crash. This reveals how the system performs through <strong>real-world-like variability</strong> versus the idealised constant-return assumption.</div>',unsafe_allow_html=True)

        seed=st.number_input('Random Seed (change to regenerate)',value=42,step=1)
        feat=e.stress(seed,1)[0]; ar=feat['ar']; months=np.arange(1,p.N+1)

        st.markdown('### Year-by-Year Returns & Performance')
        ytbl=[]
        for yr in range(p.horizon):
            idx=(yr+1)*12-1
            ytbl.append({'Year':yr+1,'Equity Return':fp(ar[yr]),'NCD System':fs(feat['np_'][idx]),'Pure SIP':fs(feat['sp'][idx]),'Advantage':fs(feat['np_'][idx]-feat['sp'][idx])})
        st.dataframe(pd.DataFrame(ytbl),use_container_width=True,hide_index=True)

        cl,cr=st.columns(2)
        with cl:
            cb=[C['grn'] if r>=0 else C['red'] for r in ar]
            fb=go.Figure(go.Bar(x=list(range(1,p.horizon+1)),y=ar*100,marker_color=cb,text=[fp(r) for r in ar],textposition='outside',textfont=dict(size=9,color=C['tx2'])))
            fb.update_layout(xaxis_title='Year',yaxis_title='Return (%)',yaxis=dict(zeroline=True,zerolinecolor=C['mu']))
            st.plotly_chart(_lo(fb,'Random Annual Returns',380),use_container_width=True)
        with cr:
            fpth=go.Figure()
            fpth.add_trace(go.Scatter(x=months,y=feat['np_'],name='NCD System',line=dict(color=C['gold'],width=2.5)))
            fpth.add_trace(go.Scatter(x=months,y=feat['sp'],name='Pure SIP',line=dict(color=C['red'],width=1.8,dash='dash')))
            fpth.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
            st.plotly_chart(_lo(fpth,'Portfolio Path',380),use_container_width=True)

        st.markdown('### This Path\'s Outcome')
        m1,m2,m3,m4=st.columns(4)
        m1.metric('NCD Total',fs(feat['nt']),fx(feat['nt']/p.principal))
        m2.metric('NCD Net',fs(feat['nn']),fp((max(feat['nn'],1)/p.principal)**(1/p.horizon)-1)+' CAGR')
        m3.metric('SIP Net',fs(feat['sn']))
        m4.metric('Advantage',fs(feat['nn']-feat['sn']),'NCD - SIP')

        st.markdown('### 25 Random Paths')
        multi=e.stress(seed*1000,25); fm=go.Figure(); na=[]; sa=[]
        for i,r in enumerate(multi):
            na.append(r['nn']); sa.append(r['sn'])
            fm.add_trace(go.Scatter(x=months,y=r['np_'],showlegend=(i==0),name='NCD Paths' if i==0 else None,legendgroup='n',line=dict(color=C['gold'],width=.8),opacity=.4))
            fm.add_trace(go.Scatter(x=months,y=r['sp'],showlegend=(i==0),name='SIP Paths' if i==0 else None,legendgroup='s',line=dict(color=C['red'],width=.6),opacity=.25))
        fm.add_trace(go.Scatter(x=nd['month'],y=nd['total'],name='Deterministic',line=dict(color='white',width=2,dash='dot')))
        fm.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
        st.plotly_chart(_lo(fm,'25 Paths — NCD (gold) vs SIP (red) vs Deterministic (white)',480),use_container_width=True)

        na_=np.array(na); sa_=np.array(sa)
        s1,s2,s3,s4=st.columns(4)
        s1.metric('NCD Mean Net',fs(na_.mean()))
        s2.metric('NCD Range',f'{fs(na_.min())} – {fs(na_.max())}')
        s3.metric('SIP Mean Net',fs(sa_.mean()))
        s4.metric('NCD Wins',f'{np.sum(na_>sa_)}/25 paths')

    with t4:
        st.markdown('## Monte Carlo Simulation')
        st.markdown(f'<div class="adam"><span class="tag">Adam · GBM</span><code>dS = μ·S·dt + σ·S·dW</code> | σ = {fp(EQUITY_VOL,0)} | {p.mc_paths:,} paired paths | Drift-corrected for Jensen\'s inequality</div>',unsafe_allow_html=True)
        with st.spinner(f'{p.mc_paths:,} simulations...'):
            mc=e.mc(); met=e.met(nd,si,fd,mc)

        cl,cr=st.columns(2)
        for col,key,title in [(cl,'pp','NCD System'),(cr,'sp','Pure SIP')]:
            with col:
                fig=go.Figure(); m=np.arange(1,p.N+1); d=mc[key]; clr=C['gold'] if key=='pp' else C['red']
                rgb=','.join(str(int(clr.lstrip('#')[i:i+2],16)) for i in (0,2,4))
                fig.add_trace(go.Scatter(x=np.concatenate([m,m[::-1]]),y=np.concatenate([d[95],d[5][::-1]]),fill='toself',fillcolor=f'rgba({rgb},.06)',line=dict(color='rgba(0,0,0,0)'),name='5th–95th',hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=np.concatenate([m,m[::-1]]),y=np.concatenate([d[75],d[25][::-1]]),fill='toself',fillcolor=f'rgba({rgb},.12)',line=dict(color='rgba(0,0,0,0)'),name='25th–75th',hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=m,y=d[50],name='Median',line=dict(color=clr,width=2.5)))
                if key=='pp': fig.add_hline(y=p.principal,line_dash='dot',line_color=C['mu'])
                st.plotly_chart(_lo(fig,title,420),use_container_width=True)

        fh=go.Figure()
        fh.add_trace(go.Histogram(x=mc['fn']/1e5,nbinsx=70,name='NCD Net',marker_color=C['gold'],opacity=.6))
        fh.add_trace(go.Histogram(x=mc['sn']/1e5,nbinsx=70,name='SIP Net',marker_color=C['red'],opacity=.4))
        fh.update_layout(barmode='overlay',xaxis_title='Net Wealth (₹ Lakhs)',yaxis_title='Frequency')
        st.plotly_chart(_lo(fh,'Terminal Distribution — Net of Tax',400),use_container_width=True)

        st.markdown('### Percentile Comparison (Net)')
        ptbl=[]
        for k in [5,10,25,50,75,90,95]:
            nv=mc['pn'][k]; sv=mc['spn'][k]
            ptbl.append({'Pctl':f'{k}th','NCD':fs(nv),'SIP':fs(sv),'Advantage':fs(nv-sv)})
        st.dataframe(pd.DataFrame(ptbl),use_container_width=True,hide_index=True)

        st.markdown('### Risk Metrics')
        r1,r2,r3,r4=st.columns(4)
        r1.metric('Sharpe',f'{met["sh"]:.3f}'); r2.metric('Sortino',f'{met["so"]:.3f}')
        r3.metric('Omega',f'{met["om"]:.3f}' if met['om']<1000 else '∞')
        r4.metric('Calmar',f'{met["ca"]:.3f}' if met['ca']<1000 else '∞')
        r5,r6,r7,r8=st.columns(4)
        r5.metric('VaR (5%)',fs(met['v5'])); r6.metric('CVaR (5%)',fs(met['cv5']))
        r7.metric('P(Capital Safe)',fp(met['pc'],1)); r8.metric('Median Max DD',fp(met['md'],1))
        r9,r10,r11,r12=st.columns(4)
        r9.metric('P(NCD > SIP)',fp(met['pb'],1)); r10.metric('P(2× Capital)',fp(met['p2'],1))
        r11.metric('Break-Even Eq.',fp(met['be'])); r12.metric('Mean CAGR (MC)',fp(met['mc_']))

        fd_=go.Figure(go.Histogram(x=mc['dd']*100,nbinsx=60,marker_color=C['red'],opacity=.6))
        fd_.add_vline(x=met['md']*100,line_dash='dash',line_color=C['gold'],annotation_text=f'Median: {fp(met["md"],1)}')
        fd_.update_layout(xaxis_title='Max Drawdown (%)',yaxis_title='Freq')
        st.plotly_chart(_lo(fd_,'Max Drawdown Distribution',350),use_container_width=True)

    with t5:
        st.markdown('## Tax Analysis & System Comparison')
        st.markdown(f'<div class="adam"><span class="tag">Adam · Tax</span><strong>NCD Interest:</strong> Slab {fp(p.tax_slab,0)} + Cess {fp(p.cess,0)} = {fp(p.eff_tax,1)}. TDS {fp(p.tds_rate,0)}.<br><strong>LTCG:</strong> {fp(p.ltcg_rate,1)} on gains > {fi(p.ltcg_exempt)}/yr. <strong>STCG:</strong> {fp(p.stcg_rate,0)}. (Budget 2024)</div>',unsafe_allow_html=True)

        ai=p.principal*p.ncd_rate*p.horizon
        labels=['Interest\nIncome','Equity\nGains','Interest\nTax','LTCG\nTax','Net\nWealth']
        vals=[ai,met['ng'],-met['nti'],-met['nl'],met['nn']]
        meas=['relative','relative','relative','relative','total']
        fw=go.Figure(go.Waterfall(x=labels,y=vals,measure=meas,connector=dict(line=dict(color=C['mu'],width=1)),decreasing=dict(marker=dict(color=C['red'])),increasing=dict(marker=dict(color=C['grn'])),totals=dict(marker=dict(color=C['gold'])),textposition='outside',text=[fs(abs(v)) for v in vals],textfont=dict(color=C['tx2'],size=9)))
        st.plotly_chart(_lo(fw,'NCD System — Income to Net Wealth',420),use_container_width=True)

        st.markdown('### Tax Breakdown')
        ttbl=pd.DataFrame({'Component':['Interest Income','Tax on Interest','TDS Deducted','Equity Gains','LTCG Tax','Total Tax','Total Wealth','Net Wealth'],
            'NCD System':[fi(ai),fi(met['nti']),fi(p.gross_m*p.tds_rate*p.N),fi(met['ng']),fi(met['nl']),fi(met['ntt']),fi(met['nt']),fi(met['nn'])],
            'Pure SIP':['—','—','—',fi(met['sg']),fi(met['sl']),fi(met['sl']),fi(met['st_']),fi(met['sn'])],
            f'FD+Eq ({fp(p.fd_rate,0)})':[fi(p.principal*p.fd_rate*p.horizon),fi(p.principal*p.fd_rate*p.horizon*p.eff_tax),fi(p.principal*p.fd_rate/12*p.tds_rate*p.N),fi(ff['gains']),fi(ff['ltcg']),fi(met['ft_']-met['fn_']),fi(met['ft_']),fi(met['fn_'])]})
        st.dataframe(ttbl,use_container_width=True,hide_index=True)

        td=met['ntt']-met['sl']
        st.markdown(f'<div class="th"><strong>Tax cost of yield arbitrage:</strong> NCD system pays {fs(met["ntt"])} vs SIP\'s {fs(met["sl"])} — extra {fs(td)}. This is offset by the {fs(p.principal)} principal returned, yielding net advantage of <strong>{fs(met["adv"])}</strong>.<br><br><strong>LTCG Harvesting:</strong> Annual redemption within {fi(p.ltcg_exempt)} exemption saves up to {fi(p.ltcg_exempt*p.ltcg_rate)}/yr — potential {fi(p.ltcg_exempt*p.ltcg_rate*p.horizon)} over {p.horizon} years.</div>',unsafe_allow_html=True)

        st.markdown('### Risk Decomposition')
        for nm,sev,cl,desc in [('Credit Risk','CRITICAL',C['red'],'NCD issuer default → principal loss. Diversify 5–10 issuers, AA- min, secured, quarterly monitoring.'),('Equity Drawdown','HIGH',C['org'],f'Median max DD {fp(met["md"],1)} in MC. DCA from interest buys more at lower prices.'),('Interest Rate','LOW (HTM)',C['grn'],'Fixed coupon. Zero impact if held to maturity.'),('Inflation','MEDIUM',C['gold'],f'At {fp(p.inflation,0)}, real coupon declines. Equity hedges long-term.'),('Tax Policy','MEDIUM',C['gold'],'Budget 2024 raised LTCG to 12.5%. Further changes possible.'),('Reinvestment','LOW–MED',C['cyn'],'At maturity, comparable yields may be unavailable. Ladder strategy recommended.')]:
            st.markdown(f'<div class="rc" style="border-left:3px solid {cl}"><div class="rt" style="color:{cl}">{nm} — {sev}</div><div class="rb">{desc}</div></div>',unsafe_allow_html=True)

    with t6:
        st.markdown('## Adam Deep Literature')
        ntc=p.ncd_rate*p.horizon*p.eff_tax
        st.markdown(f'<div class="adam"><span class="tag">Theorem 1 · Yield Arbitrage</span><code>Advantage = P·(1 - r_d·T·τ) = {fi(p.principal)}×(1-{ntc:.4f}) = {fi(p.principal*(1-ntc))}</code><br>Advantageous when r_d·T·τ < 1. Current: {ntc:.4f} < 1 ✓. Actual: <code>{fi(met["adv"])}</code> ∎</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="adam"><span class="tag">Ordinary Annuity</span><code>V(N)=S·[(1+r)^N-1]/r</code> | S={fi(p.net_m,True)} r={p.mr:.10f} N={p.N}<br>CF: <code>{fi(cfv,True)}</code> Sim: <code>{fi(nf["mf"],True)}</code> Δ={fi(delta,True)} ✓</div>',unsafe_allow_html=True)
        mu_adj=np.log(1+p.net_eq)/12-EQUITY_VOL**2/24
        st.markdown(f'<div class="adam"><span class="tag">GBM Drift</span><code>μ_adj=ln(1+r)/12-σ²/24={mu_adj:.8f}</code> σ_m={EQUITY_VOL/np.sqrt(12):.8f}<br>MC Median vs Det: {fs(mc["pct"][50])} vs {fs(nf["total"])} (Δ {abs(mc["pct"][50]-nf["total"])/nf["total"]:.2%})</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="adam"><span class="tag">Net Wealth</span><code>W=P+V(N)-P·r_d·T·τ-max(0,V-S·N-E)·τ_l</code><br>={fi(p.principal)}+{fi(nf["mf"])}-{fi(met["nti"])}-{fi(met["nl"])}=<code>{fi(met["nn"])}</code><br>Post-Tax CAGR: <code>{fp(met["ncp"])}</code></div>',unsafe_allow_html=True)
        sk=stats.skew(mc['ft']); ku=stats.kurtosis(mc['ft'])
        st.markdown(f'<div class="adam"><span class="tag">Distribution</span>Mean: {fs(mc["ft"].mean())} | Std: {fs(mc["ft"].std())} | Skew: {sk:.4f} (positive→favourable) | Kurt: {ku:.4f}<br>P(NCD>SIP): {fp(met["pb"])} | P(Capital): {fp(met["pc"])} | P(2×): {fp(met["p2"])}</div>',unsafe_allow_html=True)

    st.markdown(f'<div style="text-align:center;padding:24px 0 12px;margin-top:32px;border-top:1px solid {C["gold_b"]}"><div style="font-family:Inter;font-weight:800;font-size:.82rem;color:{C["gold"]};letter-spacing:.06em">◈ वृद्धि VRIDDHI</div><div style="font-family:JetBrains Mono;font-size:.58rem;color:{C["mu"]};margin-top:3px;letter-spacing:.1em">HEMREK CAPITAL — ADAM DEEP LITERATURE ENGINE</div></div>',unsafe_allow_html=True)

if __name__=='__main__': main()
