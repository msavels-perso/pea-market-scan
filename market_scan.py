#!/usr/bin/env python3
import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import yfinance as yf

CAC40 = {
    "AI.PA":"Air Liquide","AIR.PA":"Airbus","ALO.PA":"Alstom","ATO.PA":"Atos",
    "BN.PA":"Danone","BNP.PA":"BNP Paribas","CA.PA":"Carrefour","CAP.PA":"Capgemini",
    "CS.PA":"AXA","DG.PA":"Vinci","DSY.PA":"Dassault Systèmes","EL.PA":"EssilorLuxottica",
    "ENGI.PA":"Engie","ERF.PA":"Eurofins Scientific","GLE.PA":"Société Générale",
    "HO.PA":"Thales","KER.PA":"Kering","LR.PA":"Legrand","MC.PA":"LVMH",
    "ML.PA":"Michelin","ORA.PA":"Orange","PUB.PA":"Publicis Groupe","RI.PA":"Pernod Ricard",
    "RMS.PA":"Hermès","RNO.PA":"Renault","SAF.PA":"Safran","SAN.PA":"Sanofi",
    "SGO.PA":"Saint-Gobain","STLAP.PA":"Stellantis","STM.PA":"STMicroelectronics",
    "SU.PA":"Schneider Electric","TTE.PA":"TotalEnergies","URW.PA":"Unibail-Rodamco",
    "VIE.PA":"Veolia","ACA.PA":"Crédit Agricole","BVI.PA":"Bureau Veritas",
    "RCO.PA":"Remy Cointreau","NK.PA":"Imerys","SOP.PA":"Sopra Steria",
}
DAX = {
    "ADS.DE":"Adidas","ALV.DE":"Allianz","BAYN.DE":"Bayer","BMW.DE":"BMW",
    "BAS.DE":"BASF","DB1.DE":"Deutsche Börse","DBK.DE":"Deutsche Bank",
    "DPW.DE":"Deutsche Post","DTE.DE":"Deutsche Telekom","EOAN.DE":"E.ON",
    "FRE.DE":"Fresenius","HEI.DE":"HeidelbergCement","HEN3.DE":"Henkel",
    "IFX.DE":"Infineon Technologies","LIN.DE":"Linde","MRK.DE":"Merck KGaA",
    "MUV2.DE":"Munich Re","RWE.DE":"RWE","SAP.DE":"SAP","SIE.DE":"Siemens",
    "SRT3.DE":"Sartorius","VOW3.DE":"Volkswagen","ZAL.DE":"Zalando",
}
OTHER_EU = {
    "ASML.AS":"ASML","INGA.AS":"ING Group","MT.AS":"ArcelorMittal","PHIA.AS":"Philips",
    "REN.AS":"RELX","UNA.AS":"Unilever","WKL.AS":"Wolters Kluwer",
    "NOVO-B.CO":"Novo Nordisk (DK)","ITX.MC":"Inditex (ES)","SAN.MC":"Banco Santander (ES)",
    "IBE.MC":"Iberdrola (ES)","ENI.MI":"ENI (IT)","RACE.MI":"Ferrari (IT)",
}
PEA_ETFS = {
    "PAASI.PA":"Amundi PEA Emerging Asia ESG","PAEEM.PA":"Amundi PEA Emerging Markets",
    "PINR.PA":"Amundi PEA MSCI India","PAEJ.PA":"Amundi PEA Japan",
    "PTPXE.PA":"Amundi PEA Topix","DCAM.PA":"Amundi MSCI Europe",
    "ESE.PA":"BNP Paribas S&P 500 (PEA)","WPEA.PA":"iShares MSCI World Swap PEA",
    "ANX.PA":"Amundi Nasdaq-100 (PEA)","RS2K.PA":"Amundi Russell 2000 (PEA)",
}
NON_PEA = {
    "NVDA":"NVIDIA","AAPL":"Apple","MSFT":"Microsoft","AMZN":"Amazon",
    "META":"Meta","GOOGL":"Alphabet","TSM":"TSMC (Taiwan)","AVGO":"Broadcom",
    "005930.KS":"Samsung (Corée)","7203.T":"Toyota (Japon)","6758.T":"Sony (Japon)",
    "9984.T":"SoftBank (Japon)","BRK-B":"Berkshire Hathaway","XOM":"ExxonMobil",
    "JPM":"JPMorgan Chase",
}
INDICES = {
    "^FCHI":"CAC 40","^STOXX50E":"Euro Stoxx 50","^GDAXI":"DAX 40",
    "^AEX":"AEX (Amsterdam)","^GSPC":"S&P 500","^IXIC":"NASDAQ",
    "^N225":"Nikkei 225","^HSI":"Hang Seng",
}

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, sig=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line = ef - es
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal

def bollinger(close, period=20, nb=2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + nb*std; lower = mid - nb*std
    bw = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, mid, bw, pct_b

def sma(close, p): return close.rolling(p).mean()

def analyze(ticker, df):
    if df is None or len(df) < 60: return None
    close = df["Close"].dropna()
    volume = df["Volume"].dropna() if "Volume" in df.columns else None
    if len(close) < 60: return None
    try:
        r = rsi(close)
        ml, ms, mh = macd(close)
        _, _, _, bw, pct_b = bollinger(close)
        s20, s50, s200 = sma(close,20), sma(close,50), sma(close,200)
        cr = float(r.iloc[-1])
        ch = float(mh.iloc[-1]); ph = float(mh.iloc[-2]) if len(mh)>1 else 0
        cp = float(close.iloc[-1])
        cs20 = float(s20.iloc[-1]) if not np.isnan(s20.iloc[-1]) else None
        cs50 = float(s50.iloc[-1]) if not np.isnan(s50.iloc[-1]) else None
        cs200 = float(s200.iloc[-1]) if not np.isnan(s200.iloc[-1]) else None
        cbw = float(bw.iloc[-1]); pbw = float(bw.iloc[-5]) if len(bw)>5 else cbw
        squeeze = cbw < pbw*0.85
        cpb = float(pct_b.iloc[-1]) if not np.isnan(pct_b.iloc[-1]) else 0.5
        vr = None
        if volume is not None and len(volume)>=20:
            v5 = float(volume.iloc[-5:].mean()); v20 = float(volume.iloc[-20:].mean())
            if v20>0: vr = v5/v20
        bc = ch>0 and ph<=0
        a20 = cp>cs20 if cs20 else None
        a50 = cp>cs50 if cs50 else None
        a200 = cp>cs200 if cs200 else None
        c1d = float((close.iloc[-1]/close.iloc[-2]-1)*100) if len(close)>=2 else 0
        c1m = float((close.iloc[-1]/close.iloc[-22]-1)*100) if len(close)>=22 else 0
        os = 0
        if cr<25: os+=3
        elif cr<35: os+=2
        elif cr<45: os+=1
        if bc: os+=2
        if a200: os+=1
        if vr and vr>=1.5: os+=1
        if cpb<0.2: os+=1
        ts = 0
        if a20: ts+=1
        if a50: ts+=1
        if a200: ts+=2
        if float(ml.iloc[-1])>float(ms.iloc[-1]): ts+=1
        if 45<=cr<=68: ts+=1
        return {"rsi":cr,"macd":float(ml.iloc[-1]),"signal":float(ms.iloc[-1]),"hist":ch,
                "bull_cross":bc,"above_sma20":a20,"above_sma50":a50,"above_sma200":a200,
                "vol_ratio":vr,"squeeze":squeeze,"pct_b":cpb,
                "oversold_score":os,"trend_score":ts,"overbought":cr>72,
                "price":cp,"change_1d":c1d,"change_1mo":c1m}
    except: return None

def dl(tickers, period="14mo", label=""):
    if not tickers: return {}
    print(f"  {label} ({len(tickers)})...", end=" ", flush=True)
    try:
        raw = yf.download(list(tickers), period=period, auto_adjust=True,
                          group_by="ticker", threads=True, progress=False)
    except Exception as e: print(f"ERR:{e}"); return {}
    result = {}
    for t in tickers:
        try:
            df = raw if len(tickers)==1 else (raw[t] if t in raw.columns.get_level_values(0) else None)
            if df is not None and not df.empty and len(df)>=30: result[t]=df
        except: pass
    print(f"{len(result)}/{len(tickers)} OK")
    return result

def get_idx(ticker):
    try:
        df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1]), float((df["Close"].iloc[-1]/df["Close"].iloc[-2]-1)*100)
    except: pass
    return None, None

def fmts(s):
    parts = [f"RSI {s['rsi']:.0f}"]
    if s["bull_cross"]: parts.append("MACD↑ croisement haussier")
    elif s["macd"]>s["signal"]: parts.append("MACD > signal")
    else: parts.append("MACD < signal")
    if s["above_sma200"] is True: parts.append("▲SMA200")
    elif s["above_sma200"] is False: parts.append("▼SMA200")
    if s["vol_ratio"] and s["vol_ratio"]>=1.5: parts.append(f"Vol×{s['vol_ratio']:.1f}")
    if s["squeeze"]: parts.append("BB squeeze")
    return " | ".join(parts)

def fmtp(s): return f"{s['change_1d']:+.1f}%/j  {s['change_1mo']:+.1f}%/mois"

W=70
def sec(title, c="─"): print(); print(c*W); print(f"  {title}"); print(c*W)
def row(name, ticker, s, extra=""):
    print(f"  {(name+' ('+ticker+')'):<44} {fmtp(s)}")
    print(f"    {fmts(s)}")
    if extra: print(f"    {extra}")

def main():
    now = datetime.now(timezone.utc)
    print("="*W)
    print(f"  SCAN OPPORTUNITÉS MARCHÉ — {now.strftime('%A %d/%m/%Y — %H:%M UTC')}")
    print(f"  Univers : CAC40 + DAX + EU + ETFs PEA + Hors-PEA ({sum(map(len,[CAC40,DAX,OTHER_EU,PEA_ETFS,NON_PEA]))} valeurs)")
    print("="*W)

    sec("CONTEXTE INDICES", "─")
    for t, n in INDICES.items():
        p, c = get_idx(t)
        if p: print(f"  {n:<28} {p:>10,.2f}   {'▲' if c>=0 else '▼'}{abs(c):.2f}%")
        else: print(f"  {n:<28} {'n/d':>10}")

    print()
    all_pea = {**CAC40,**DAX,**OTHER_EU,**PEA_ETFS}
    dpea  = dl(list(CAC40.keys()),    label="CAC 40")
    ddax  = dl(list(DAX.keys()),      label="DAX")
    deu   = dl(list(OTHER_EU.keys()), label="Autres EU")
    detf  = dl(list(PEA_ETFS.keys()), label="ETFs PEA")
    dnpea = dl(list(NON_PEA.keys()),  label="Hors-PEA")
    all_data = {**dpea,**ddax,**deu,**detf}

    sig, snp = {}, {}
    for t, df in all_data.items():
        s = analyze(t, df)
        if s: sig[t] = {"name": all_pea.get(t,t), **s}
    for t, df in dnpea.items():
        s = analyze(t, df)
        if s: snp[t] = {"name": NON_PEA.get(t,t), **s}

    ca = {t:s for t,s in sig.items() if s["oversold_score"]>=4 and s["above_sma200"] is not False and not s["overbought"]}
    ca = dict(sorted(ca.items(), key=lambda x: x[1]["oversold_score"], reverse=True))
    sec("A. SIGNAUX D'ACHAT — convergence multi-indicateurs (éligibles PEA)","=")
    if ca:
        for t,s in ca.items(): row(s["name"],t,s,f"[Score: {s['oversold_score']}/8]")
    else: print("  Aucun signal fort détecté aujourd'hui.")

    cb = {t:s for t,s in sig.items() if 2<=s["oversold_score"]<4 and s["above_sma200"] is not False and not s["overbought"] and t not in ca}
    cb = dict(sorted(cb.items(), key=lambda x: x[1]["oversold_score"], reverse=True)[:12])
    sec("B. À SURVEILLER — RSI en zone de retournement ou MACD s'amorce","─")
    if cb:
        for t,s in cb.items(): row(s["name"],t,s)
    else: print("  Aucune valeur en zone de surveillance.")

    cc = {t:s for t,s in sig.items() if s["trend_score"]>=4 and s["oversold_score"]<=2 and not s["overbought"]}
    cc = dict(sorted(cc.items(), key=lambda x: x[1]["trend_score"], reverse=True)[:10])
    sec("C. TENDANCE HAUSSIÈRE CONFIRMÉE — momentum solide","─")
    if cc:
        for t,s in cc.items(): row(s["name"],t,s,f"[Trend: {s['trend_score']}/6]")
    else: print("  Aucune valeur en tendance forte.")

    cd = {t:s for t,s in sig.items() if s["overbought"]}
    cd = dict(sorted(cd.items(), key=lambda x: x[1]["rsi"], reverse=True)[:8])
    sec("D. SURACHAT — prudence / prise de bénéfices","─")
    if cd:
        for t,s in cd.items(): row(s["name"],t,s)
    else: print("  Aucune valeur en surachat significatif.")

    sq = {t:s for t,s in sig.items() if s.get("squeeze") and not s["overbought"]}
    if sq:
        sec("⚡ COMPRESSIONS BOLLINGER — rupture imminente","─")
        for t,s in list(sq.items())[:8]:
            print(f"  {s['name']:<38} ({t})  RSI {s['rsi']:.0f}  {fmtp(s)}")

    sec("HORS PEA — Compte-titres ordinaire uniquement","=")
    ha = {t:s for t,s in snp.items() if s["oversold_score"]>=3 and not s["overbought"]}
    hc = {t:s for t,s in snp.items() if s["trend_score"]>=4 and not s["overbought"] and t not in ha}
    if ha:
        print("\n  ▶ Signaux d'achat / rebond :")
        for t,s in sorted(ha.items(),key=lambda x:x[1]["oversold_score"],reverse=True): row(s["name"],t,s)
    if hc:
        print("\n  ▶ Tendances haussières :")
        for t,s in sorted(hc.items(),key=lambda x:x[1]["trend_score"],reverse=True): row(s["name"],t,s)
    if not ha and not hc: print("  Aucun signal notable hors-PEA.")

    sec("RÉSUMÉ","=")
    print(f"  Valeurs analysées (PEA) : {len(sig)}")
    print(f"  A (achat fort) : {len(ca)}  |  B (surveiller) : {len(cb)}  |  C (tendance) : {len(cc)}  |  D (surachat) : {len(cd)}")
    print(f"  Compressions BB : {len(sq)}  |  Hors-PEA : {len(snp)}")
    print("="*W)

if __name__ == "__main__":
    main()
