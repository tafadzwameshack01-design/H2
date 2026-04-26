"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ZEUS ⚡ NEURAL FOOTBALL INTELLIGENCE SYSTEM v5.0                           ║
║  FOCUS: Full-Time UNDER 1.5 · Half-Time OVER/UNDER 1.5                     ║
║  Dixon-Coles + XGBoost + ESPN + TheSportsDB + The Odds API                 ║
║  Pipeline Health · Portfolio Kelly · Injury & News Intel                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st

st.set_page_config(
    page_title="ZEUS ⚡ v5 Under 1.5 & HT Specialist",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "ZEUS Neural Football AI v5.0 — FT Under 1.5 · HT Over/Under 1.5 Specialist"},
)

# ─── Standard library ─────────────────────────────────────────────────────────
import json, math, time, hashlib, sqlite3, logging, random, pickle, os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path

# ─── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import pytz
from scipy.optimize import minimize, brentq
from scipy.stats import poisson as sp_poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
import shap

try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_OK = True
except ImportError:
    _AUTOREFRESH_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("zeus_v4")

# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
APP_VERSION           = "5.0.0"
ESPN_SOCCER           = "https://site.api.espn.com/apis/site/v2/sports/soccer"
TSDB_BASE             = "https://www.thesportsdb.com/api/v1/json/3"
CLUBELO_BASE          = "https://api.clubelo.com"
ODDS_API_BASE         = "https://api.the-odds-api.com/v4"
# Hardcoded fallback API key (user-provided)
_HARDCODED_ODDS_KEY   = "7a0f07728cacede51d6f80a5f1ec0086"

UTC                   = pytz.UTC
WINDOW_HOURS          = 6
MIN_GAMES             = 5
HISTORY_GAMES         = 38
TOP_N                 = 7
LEARNING_RATE         = 0.004

DB_PATH               = Path("/tmp/zeus_v4.db")

DC_MIN_LEAGUE_MATCHES = 150
DC_MIN_TEAM_MATCHES   = 5
XGB_MIN_TRAIN         = 300
XGB_MIN_CALIB         = 80
XGB_RETRAIN_THRESHOLD = 50
KELLY_FRACTION        = 0.25
KELLY_MAX_STAKE       = 0.05
MIN_ROI_SAMPLE        = 100
ODDS_AGE_MAX_MIN      = 120.0

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# ─── Bet type registry — ZEUS v5: FT Under 1.5 · HT Over/Under 1.5 ONLY ───
BET_TYPES: dict[str, dict] = {
    "FT_UNDER_15":  {"label": "FT UNDER 1.5 Goals", "line": 1.5, "gate": 62.0, "css": "under15", "emoji": "🔒"},
    "HT_UNDER_15":  {"label": "HT UNDER 0.5 Goals", "line": 0.5, "gate": 62.0, "css": "htunder", "emoji": "🕐"},
    "HT_OVER_15":   {"label": "HT OVER 0.5 Goals",  "line": 0.5, "gate": 62.0, "css": "htover",  "emoji": "⚡"},
}

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    # Full-time Under 1.5: heavily weight defensive stats, clean sheets, HT patterns
    "FT_UNDER_15": {"poisson_p": 0.30, "hist_under15": 0.25, "cs_rate": 0.18, "ht_under_rate": 0.15, "h2h": 0.07, "form": 0.05},
    # HT Under 0.5 (no goal in first half): low-scoring first halves, defensive teams
    "HT_UNDER_15": {"ht_poisson": 0.32, "ht_hist_under": 0.30, "cs_rate": 0.18, "form": 0.12, "h2h": 0.08},
    # HT Over 0.5 (at least 1 goal in first half): attacking teams, high-scoring first halves
    "HT_OVER_15":  {"ht_poisson": 0.32, "ht_hist_over": 0.30, "form": 0.15, "over15_rate": 0.15, "h2h": 0.08},
}

LEAGUES: list[tuple[str, str, str]] = [
    ("eng.1","Premier League","🏴󠁧󠁢󠁥󠁮󠁧󠁿"), ("eng.2","Championship","🏴󠁧󠁢󠁥󠁮󠁧󠁿"),
    ("eng.3","League One","🏴󠁧󠁢󠁥󠁮󠁧󠁿"), ("eng.4","League Two","🏴󠁧󠁢󠁥󠁮󠁧󠁿"),
    ("esp.1","La Liga","🇪🇸"), ("esp.2","Segunda División","🇪🇸"),
    ("ger.1","Bundesliga","🇩🇪"), ("ger.2","2. Bundesliga","🇩🇪"), ("ger.3","3. Liga","🇩🇪"),
    ("ita.1","Serie A","🇮🇹"), ("ita.2","Serie B","🇮🇹"),
    ("fra.1","Ligue 1","🇫🇷"), ("fra.2","Ligue 2","🇫🇷"),
    ("ned.1","Eredivisie","🇳🇱"), ("ned.2","Eerste Divisie","🇳🇱"),
    ("por.1","Primeira Liga","🇵🇹"), ("por.2","Liga Portugal 2","🇵🇹"),
    ("sco.1","Scottish Premiership","🏴󠁧󠁢󠁳󠁣󠁴󠁿"), ("sco.2","Scottish Championship","🏴󠁧󠁢󠁳󠁣󠁴󠁿"),
    ("tur.1","Süper Lig","🇹🇷"), ("tur.2","TFF First League","🇹🇷"),
    ("bel.1","Belgian Pro League","🇧🇪"), ("gre.1","Super League Greece","🇬🇷"),
    ("ukr.1","Ukrainian Premier","🇺🇦"), ("den.1","Superligaen","🇩🇰"),
    ("swe.1","Allsvenskan","🇸🇪"), ("nor.1","Eliteserien","🇳🇴"),
    ("aut.1","Austrian Bundesliga","🇦🇹"), ("sui.1","Swiss Super League","🇨🇭"),
    ("cze.1","Czech First League","🇨🇿"), ("pol.1","Ekstraklasa","🇵🇱"),
    ("rou.1","Liga 1 Romania","🇷🇴"), ("srb.1","Serbian SuperLiga","🇷🇸"),
    ("hun.1","OTP Bank Liga","🇭🇺"), ("bul.1","First Professional League","🇧🇬"),
    ("cro.1","HNL Croatia","🇭🇷"), ("svk.1","Fortuna Liga Slovakia","🇸🇰"),
    ("fin.1","Veikkausliiga","🇫🇮"), ("isr.1","Israeli Premier","🇮🇱"),
    ("rus.1","Russian Premier","🇷🇺"),
    ("usa.1","MLS","🇺🇸"), ("usa.2","USL Championship","🇺🇸"),
    ("mex.1","Liga MX","🇲🇽"), ("mex.2","Ascenso MX","🇲🇽"),
    ("bra.1","Brasileirão","🇧🇷"), ("bra.2","Série B","🇧🇷"),
    ("arg.1","Primera División","🇦🇷"), ("col.1","Liga Betplay","🇨🇴"),
    ("chi.1","Primera Chile","🇨🇱"), ("ecu.1","Liga Pro Ecuador","🇪🇨"),
    ("per.1","Liga 1 Peru","🇵🇪"), ("uru.1","Uruguay Primera","🇺🇾"),
    ("ven.1","Liga Futve","🇻🇪"), ("par.1","División Profesional","🇵🇾"),
    ("jpn.1","J1 League","🇯🇵"), ("jpn.2","J2 League","🇯🇵"),
    ("kor.1","K League 1","🇰🇷"), ("chn.1","Chinese Super League","🇨🇳"),
    ("aus.1","A-League","🇦🇺"), ("ind.1","Indian Super League","🇮🇳"),
    ("tha.1","Thai League 1","🇹🇭"), ("mys.1","Super League Malaysia","🇲🇾"),
    ("sau.1","Saudi Pro League","🇸🇦"), ("uae.1","UAE Pro League","🇦🇪"),
    ("egy.1","Egyptian Premier","🇪🇬"), ("rsa.1","PSL South Africa","🇿🇦"),
    ("mar.1","Botola Pro Morocco","🇲🇦"), ("nga.1","NPFL Nigeria","🇳🇬"),
    ("qat.1","Qatar Stars League","🇶🇦"),
    ("uefa.champions","Champions League","🏆"), ("uefa.europa","Europa League","🟠"),
    ("uefa.europaconference","Conference League","🟢"),
    ("conmebol.libertadores","Copa Libertadores","🏆"),
    ("concacaf.champions","CONCACAF Champions","🌎"),
]

FEATURE_COLS: list[str] = [
    "home_avg_scored", "home_avg_conceded",
    "away_avg_scored", "away_avg_conceded",
    "home_under15_rate", "away_under15_rate",
    "home_under15_home", "away_under15_away",
    "home_cs_rate", "away_cs_rate",
    "home_cs_home", "away_cs_away",
    "h2h_under15",
    "poisson_under15",
    "home_form3", "away_form3",
    "home_n_clipped", "away_n_clipped",
    # NEW: halftime features
    "home_ht_under_rate", "away_ht_under_rate",
    "home_ht_over_rate",  "away_ht_over_rate",
    "home_ht_avg_goals",  "away_ht_avg_goals",
]

TIMEZONE_OPTIONS = [
    "UTC", "Africa/Johannesburg", "Europe/London", "Europe/Paris",
    "Europe/Berlin", "Europe/Madrid", "Europe/Rome", "Europe/Moscow",
    "America/New_York", "America/Chicago", "America/Los_Angeles",
    "America/Sao_Paulo", "America/Buenos_Aires",
    "Asia/Dubai", "Asia/Riyadh", "Asia/Tokyo", "Asia/Shanghai",
    "Australia/Sydney",
]

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — STADIUM-AT-NIGHT v4 (evolved from v3)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500&display=swap');

:root {
  --bg:       #040b04;
  --surface:  #0a160a;
  --card:     #0d1b0d;
  --border:   #183018;
  --green:    #39ff14;
  --green2:   #00c853;
  --gold:     #ffb300;
  --gold2:    #ff8f00;
  --cyan:     #00e5ff;
  --red:      #ff1744;
  --purple:   #ea80fc;
  --orange:   #ff6d00;
  --blue:     #2979ff;
  --teal:     #1de9b6;
  --text:     #d4f0d4;
  --muted:    #4e724e;
}

html, body, .stApp { background: var(--bg) !important; font-family: 'Barlow', sans-serif; color: var(--text); }

.stApp::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(57,255,20,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(57,255,20,0.015) 1px, transparent 1px);
  background-size: 60px 60px;
  animation: gridMove 30s linear infinite;
  pointer-events: none; z-index: 0;
}
@keyframes gridMove { 100% { background-position: 60px 60px, 60px 60px; } }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; max-width: 1320px; position: relative; z-index: 1; }

/* ── Hero ── */
.zeus-hero { text-align: center; padding: 22px 0 8px; }
.zeus-logo {
  font-family: 'Bebas Neue', cursive; font-size: 5.5rem; letter-spacing: 12px; line-height: 1;
  background: linear-gradient(135deg, #39ff14 0%, #69ff47 40%, #ffb300 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  animation: logoGlow 4s ease-in-out infinite;
}
@keyframes logoGlow {
  0%,100% { filter: drop-shadow(0 0 8px rgba(57,255,20,0.4)); }
  50%      { filter: drop-shadow(0 0 28px rgba(57,255,20,0.9)); }
}
.zeus-tagline { font-family:'Barlow Condensed',sans-serif; font-size:0.78rem; letter-spacing:4px; text-transform:uppercase; color:var(--muted); margin-top:4px; }
.zeus-version { font-family:'Barlow Condensed',sans-serif; font-size:0.66rem; letter-spacing:3px; text-transform:uppercase; color:var(--cyan); margin-top:2px; opacity:0.75; }
.zeus-bar { width:80px; height:2px; background:linear-gradient(90deg,transparent,var(--green),transparent); margin:12px auto 0; animation:barPulse 2s ease-in-out infinite; }
@keyframes barPulse { 0%,100%{width:80px;opacity:.6;} 50%{width:200px;opacity:1;} }

/* ── Metrics ── */
.metrics-row { display:flex; gap:10px; margin:14px 0; flex-wrap:wrap; }
.metric-box { flex:1; min-width:90px; background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:12px 14px; text-align:center; transition:border-color .3s; }
.metric-box:hover { border-color:var(--green); }
.metric-val { font-family:'Bebas Neue',cursive; font-size:2rem; color:var(--green); line-height:1; display:block; }
.metric-val.gold   { color:var(--gold); }
.metric-val.cyan   { color:var(--cyan); }
.metric-val.purple { color:var(--purple); }
.metric-val.red    { color:var(--red); }
.metric-val.teal   { color:var(--teal); }
.metric-val.blue   { color:var(--blue); }
.metric-lbl { font-family:'Barlow Condensed',sans-serif; font-size:0.67rem; color:var(--muted); text-transform:uppercase; letter-spacing:1.5px; }

/* ── Scan line ── */
.scan-line { font-family:'Barlow Condensed',sans-serif; font-size:0.78rem; color:var(--green); letter-spacing:3px; text-transform:uppercase; text-align:center; padding:8px; animation:scanFade .9s ease-in-out infinite; }
@keyframes scanFade { 0%,100%{opacity:1;} 50%{opacity:0.2;} }

/* ── Pick cards ── */
.pick-card {
  background:var(--card); border:1px solid var(--border); border-radius:18px; padding:22px 26px; margin:14px 0;
  position:relative; overflow:hidden; opacity:0;
  animation:cardReveal .5s ease forwards; transition:transform .25s, box-shadow .25s;
}
.pick-card:hover { transform:translateY(-4px); box-shadow:0 14px 44px rgba(57,255,20,.14); }
.pick-card:nth-child(1){animation-delay:.04s} .pick-card:nth-child(2){animation-delay:.12s}
.pick-card:nth-child(3){animation-delay:.20s} .pick-card:nth-child(4){animation-delay:.28s}
.pick-card:nth-child(5){animation-delay:.36s} .pick-card:nth-child(6){animation-delay:.44s}
.pick-card:nth-child(7){animation-delay:.52s}
@keyframes cardReveal { from{opacity:0;transform:translateY(18px);} to{opacity:1;transform:translateY(0);} }

.pick-card.elite  { border-color:var(--gold);   background:linear-gradient(135deg,#0d1b0d 0%,#1a1400 100%); animation:cardReveal .5s ease forwards,eliteGlow 3s ease-in-out infinite; }
.pick-card.strong { border-color:var(--green2); }
.pick-card.btts   { border-color:var(--purple); }
.pick-card.result { border-color:var(--cyan);   }
.pick-card.value  { border-color:var(--teal);   background:linear-gradient(135deg,#0d1b0d 0%,#001a14 100%); animation:cardReveal .5s ease forwards,valueGlow 3s ease-in-out infinite; }
@keyframes eliteGlow { 0%,100%{box-shadow:0 0 16px rgba(255,179,0,.1);} 50%{box-shadow:0 0 44px rgba(255,179,0,.32);} }
@keyframes valueGlow { 0%,100%{box-shadow:0 0 16px rgba(29,233,182,.1);} 50%{box-shadow:0 0 44px rgba(29,233,182,.32);} }

.rank-badge { position:absolute; top:14px; right:20px; font-family:'Bebas Neue',cursive; font-size:4rem; line-height:1; color:rgba(57,255,20,.05); pointer-events:none; user-select:none; }

.card-league { font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; letter-spacing:3px; text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
.card-teams { font-family:'Bebas Neue',cursive; font-size:2rem; letter-spacing:3px; color:var(--text); line-height:1.1; margin-bottom:10px; }
.card-vs { color:var(--muted); font-size:1rem; padding:0 8px; }

.card-bet { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:1.5rem; letter-spacing:1px; margin-bottom:12px; }
.bet-over05  { color:var(--cyan);   }
.bet-over15  { color:var(--green);  }
.bet-over25  { color:var(--gold);   }
.bet-btts    { color:var(--purple); }
.bet-home    { color:var(--green2); }
.bet-away    { color:var(--orange); }
.bet-under15 { color:var(--teal);   }
.bet-htunder { color:var(--cyan);   }
.bet-htover  { color:var(--gold);   }

.conf-track { background:rgba(255,255,255,.06); border-radius:999px; height:6px; margin:8px 0 10px; overflow:hidden; }
.conf-fill  { height:100%; border-radius:999px; animation:fillBar 1.2s cubic-bezier(.22,1,.36,1) forwards; transform-origin:left; }
.conf-fill.elite  { background:linear-gradient(90deg,var(--gold2),var(--gold)); }
.conf-fill.strong { background:linear-gradient(90deg,var(--green2),var(--green)); }
.conf-fill.btts   { background:linear-gradient(90deg,#7b1fa2,var(--purple)); }
.conf-fill.result { background:linear-gradient(90deg,#006064,var(--cyan)); }
.conf-fill.value  { background:linear-gradient(90deg,#004d3d,var(--teal)); }
@keyframes fillBar { from{width:0 !important;} }

.conf-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.conf-pct { font-family:'Bebas Neue',cursive; font-size:1.6rem; letter-spacing:2px; }
.conf-pct.elite  { color:var(--gold); }
.conf-pct.strong { color:var(--green); }
.conf-pct.btts   { color:var(--purple); }
.conf-pct.result { color:var(--cyan); }
.conf-pct.value  { color:var(--teal); }

.tier-chip { font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; padding:3px 10px; border-radius:999px; }
.tier-chip.elite  { background:rgba(255,179,0,.15);   color:var(--gold);   border:1px solid rgba(255,179,0,.4); }
.tier-chip.strong { background:rgba(57,255,20,.1);    color:var(--green);  border:1px solid rgba(57,255,20,.3); }
.tier-chip.btts   { background:rgba(234,128,252,.1);  color:var(--purple); border:1px solid rgba(234,128,252,.3); }
.tier-chip.result { background:rgba(0,229,255,.1);    color:var(--cyan);   border:1px solid rgba(0,229,255,.3); }
.tier-chip.value  { background:rgba(29,233,182,.1);   color:var(--teal);   border:1px solid rgba(29,233,182,.3); }

.pills-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }
.pill { font-family:'Barlow Condensed',sans-serif; font-size:0.73rem; letter-spacing:1px; padding:3px 9px; border-radius:6px; white-space:nowrap; }
.pill-time  { background:rgba(57,255,20,.08);  color:var(--green);  border:1px solid rgba(57,255,20,.2); }
.pill-xg    { background:rgba(255,179,0,.08);  color:var(--gold);   border:1px solid rgba(255,179,0,.2); }
.pill-odds  { background:rgba(29,233,182,.08); color:var(--teal);   border:1px solid rgba(29,233,182,.2); }
.pill-edge  { background:rgba(57,255,20,.12);  color:var(--green);  border:1px solid rgba(57,255,20,.3); font-weight:700; }
.pill-kelly { background:rgba(255,179,0,.1);   color:var(--gold);   border:1px solid rgba(255,179,0,.3); }
.pill-warn  { background:rgba(255,23,68,.08);  color:var(--red);    border:1px solid rgba(255,23,68,.2); }
.pill-h2h   { background:rgba(255,64,64,.08);  color:#ff6464;       border:1px solid rgba(255,64,64,.2); }
.pill-learn { background:rgba(0,200,83,.08);   color:#00c853;       border:1px solid rgba(0,200,83,.2); }
.pill-calib { background:rgba(41,182,246,.08); color:#29b6f6;       border:1px solid rgba(41,182,246,.2); }
.pill-stale { background:rgba(255,23,68,.08);  color:var(--red);    border:1px solid rgba(255,23,68,.2); }

.ai-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-top:10px; }
.ai-factor { background:rgba(57,255,20,.04); border:1px solid rgba(57,255,20,.1); border-radius:8px; padding:6px 8px; text-align:center; }
.ai-factor-val { font-family:'Bebas Neue',cursive; font-size:1.1rem; color:var(--green); display:block; line-height:1; }
.ai-factor-val.gold   { color:var(--gold); }
.ai-factor-val.cyan   { color:var(--cyan); }
.ai-factor-val.purple { color:var(--purple); }
.ai-factor-val.teal   { color:var(--teal); }
.ai-factor-lbl { font-family:'Barlow Condensed',sans-serif; font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }

.card-reason { font-family:'Barlow',sans-serif; font-size:0.8rem; color:var(--muted); margin-top:10px; line-height:1.55; border-left:2px solid var(--border); padding-left:10px; }
.card-disclaimer { font-family:'Barlow',sans-serif; font-size:0.7rem; color:rgba(255,23,68,.6); margin-top:6px; font-style:italic; }
.countdown { font-family:'Bebas Neue',cursive; font-size:0.85rem; letter-spacing:2px; color:var(--green); }
.no-picks { text-align:center; padding:52px 24px; font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; color:var(--muted); letter-spacing:2px; }
.no-picks-icon { font-size:3rem; display:block; margin-bottom:12px; }
.learn-badge { display:inline-block; background:rgba(0,200,83,.1); border:1px solid rgba(0,200,83,.3); border-radius:8px; padding:4px 10px; font-family:'Barlow Condensed',sans-serif; font-size:0.72rem; color:#00c853; letter-spacing:2px; text-transform:uppercase; }

/* ── Health banners ── */
.health-ok   { background:rgba(57,255,20,.07); border:1px solid rgba(57,255,20,.25); border-radius:8px; padding:8px 14px; font-family:'Barlow Condensed',sans-serif; font-size:0.8rem; color:var(--green); letter-spacing:1px; margin-bottom:12px; }
.health-warn { background:rgba(255,179,0,.07); border:1px solid rgba(255,179,0,.25); border-radius:8px; padding:8px 14px; font-family:'Barlow Condensed',sans-serif; font-size:0.8rem; color:var(--gold);  letter-spacing:1px; margin-bottom:12px; }
.health-fail { background:rgba(255,23,68,.07); border:1px solid rgba(255,23,68,.25);  border-radius:8px; padding:8px 14px; font-family:'Barlow Condensed',sans-serif; font-size:0.8rem; color:var(--red);   letter-spacing:1px; margin-bottom:12px; }

hr { border-color:rgba(57,255,20,.08) !important; }
.stTabs [data-baseweb="tab-list"] { background:var(--surface); border-radius:12px; padding:4px; gap:2px; border:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius:8px; font-family:'Barlow Condensed',sans-serif; letter-spacing:1px; color:var(--muted); font-size:.9rem; }
.stTabs [aria-selected="true"] { background:rgba(57,255,20,.12) !important; color:var(--green) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALIZATION — ALL KEYS BEFORE ANY LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "pipeline_health": {},
        "shap_cache": {},
        "user_timezone": "UTC",
        "xgb_model": None,
        "xgb_model_meta": {},
        "dc_models": {},          # league_id → DixonColesResult
        "refresh_count": 0,
        "odds_quota_remaining": None,
        "iteration_log": [],
        "prev_picks": [],
        "model_initialized": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session_state()


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE — SQLite schema + helpers
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key TEXT PRIMARY KEY, data TEXT, ts REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS picks_log (
            id           TEXT PRIMARY KEY,
            match        TEXT,
            league       TEXT,
            league_id    TEXT,
            bet          TEXT,
            bet_type     TEXT DEFAULT 'OVER_25',
            xg_total     REAL,
            confidence   REAL,
            kickoff      TEXT,
            result       TEXT DEFAULT 'pending',
            home_score   INTEGER DEFAULT -1,
            away_score   INTEGER DEFAULT -1,
            factors_json TEXT DEFAULT '{}',
            logged_at    TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            match_id     TEXT PRIMARY KEY,
            league_id    TEXT,
            home_team    TEXT,
            away_team    TEXT,
            home_goals   INTEGER,
            away_goals   INTEGER,
            match_date   TEXT,
            source       TEXT DEFAULT 'espn',
            validated    INTEGER DEFAULT 0,
            created_at   TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS value_predictions (
            pred_id         TEXT PRIMARY KEY,
            match_id        TEXT,
            league_id       TEXT,
            home_team       TEXT,
            away_team       TEXT,
            kickoff_utc     TEXT,
            model_prob      REAL,
            implied_prob    REAL,
            odds_under15    REAL,
            odds_source     TEXT,
            odds_fetched_at TEXT,
            edge            REAL,
            kelly_stake     REAL,
            correlation_discount INTEGER DEFAULT 0,
            actual_outcome  INTEGER DEFAULT -1,
            pnl_units       REAL DEFAULT 0.0,
            features_json   TEXT DEFAULT '{}',
            created_at      TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_weights (
            bet_type TEXT,
            factor   TEXT,
            weight   REAL,
            wins     INTEGER DEFAULT 0,
            losses   INTEGER DEFAULT 0,
            updates  INTEGER DEFAULT 0,
            PRIMARY KEY (bet_type, factor)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            source     TEXT,
            status     TEXT,
            message    TEXT,
            rows       INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0,
            logged_at  TEXT
        )
    """)

    # Auto-migrate picks_log
    for col, defval in [
        ("bet_type", "'OVER_25'"), ("home_score", "-1"),
        ("away_score", "-1"), ("factors_json", "'{}'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE picks_log ADD COLUMN {col} TEXT DEFAULT {defval}")
        except Exception:
            pass

    conn.commit()
    _init_weights(conn)
    return conn


def _init_weights(conn: sqlite3.Connection) -> None:
    for bet_type, factors in DEFAULT_WEIGHTS.items():
        for factor, w in factors.items():
            conn.execute(
                "INSERT OR IGNORE INTO model_weights (bet_type,factor,weight) VALUES (?,?,?)",
                (bet_type, factor, w)
            )
    conn.commit()


# ── Cache helpers ──────────────────────────────────────────────────────────────
def cache_get(key: str, ttl: int) -> Optional[Any]:
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT data, ts FROM api_cache WHERE cache_key=?", (key,)
        ).fetchone()
        if row and (time.time() - row[1]) < ttl:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def cache_set(key: str, data: Any) -> None:
    try:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO api_cache VALUES (?,?,?)",
            (key, json.dumps(data, default=str), time.time())
        )
        conn.commit()
    except Exception:
        pass


# ── Weight helpers ─────────────────────────────────────────────────────────────
def get_weights(bet_type: str) -> dict[str, float]:
    conn = get_db()
    rows = conn.execute(
        "SELECT factor, weight FROM model_weights WHERE bet_type=?", (bet_type,)
    ).fetchall()
    if not rows:
        return DEFAULT_WEIGHTS.get(bet_type, {})
    w = {r[0]: r[1] for r in rows}
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w


def update_weights(bet_type: str, factors: dict[str, float], won: bool) -> None:
    conn = get_db()
    current = get_weights(bet_type)
    signal = 1.0 if won else -0.5
    new_w: dict[str, float] = {}
    for factor, val in factors.items():
        if factor not in current:
            continue
        contribution = val - 0.5
        delta = LEARNING_RATE * signal * contribution
        new_w[factor] = max(0.02, min(0.70, current[factor] + delta))
    total = sum(new_w.values())
    if total > 0:
        new_w = {k: v / total for k, v in new_w.items()}
    for factor, weight in new_w.items():
        result_col = "wins" if won else "losses"
        conn.execute(
            f"""UPDATE model_weights
                SET weight=?, {result_col}={result_col}+1, updates=updates+1
                WHERE bet_type=? AND factor=?""",
            (weight, bet_type, factor)
        )
    conn.commit()


# ── Pipeline health logger ─────────────────────────────────────────────────────
def log_pipeline(source: str, status: str, message: str, rows: int = 0, latency_ms: float = 0.0) -> None:
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO pipeline_log (source,status,message,rows,latency_ms,logged_at) VALUES (?,?,?,?,?,?)",
            (source, status, message, rows, latency_ms,
             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        )
        conn.commit()
        st.session_state["pipeline_health"][source] = {
            "status": status,
            "message": message,
            "rows": rows,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc),
        }
    except Exception:
        pass


# ── Match result storage ───────────────────────────────────────────────────────
def validate_match_result(home_goals: Any, away_goals: Any) -> tuple[bool, str]:
    if home_goals is None or away_goals is None:
        return False, "Missing goal data"
    try:
        hg, ag = int(home_goals), int(away_goals)
    except (ValueError, TypeError):
        return False, f"Non-integer goals: {home_goals}-{away_goals}"
    if hg < 0 or ag < 0:
        return False, f"Negative goals: {hg}-{ag}"
    if hg > 15 or ag > 15:
        return False, f"Implausible score: {hg}-{ag} — likely data error"
    return True, "valid"


def store_match_result(
    league_id: str, home_team: str, away_team: str,
    home_goals: int, away_goals: int, match_date: str
) -> bool:
    valid, msg = validate_match_result(home_goals, away_goals)
    if not valid:
        logger.warning("Rejected match result: %s", msg)
        return False
    match_id = hashlib.md5(
        f"{match_date[:10]}_{home_team}_{away_team}_{league_id}".encode()
    ).hexdigest()[:16]
    try:
        conn = get_db()
        conn.execute(
            """INSERT OR IGNORE INTO match_results
               (match_id,league_id,home_team,away_team,home_goals,away_goals,
                match_date,validated,created_at)
               VALUES (?,?,?,?,?,?,?,1,?)""",
            (match_id, league_id, home_team, away_team, home_goals, away_goals,
             match_date, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        )
        conn.commit()
        return True
    except Exception:
        return False


def get_training_count() -> int:
    try:
        conn = get_db()
        row = conn.execute("SELECT COUNT(*) FROM match_results WHERE validated=1").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


# ── Picks logger ───────────────────────────────────────────────────────────────
def save_pick(pick: dict) -> None:
    try:
        pid = hashlib.md5(
            f"{pick['match']}{pick['kickoff_utc']}{pick['bet_type']}".encode()
        ).hexdigest()[:12]
        conn = get_db()
        conn.execute(
            """INSERT OR IGNORE INTO picks_log
               (id,match,league,league_id,bet,bet_type,xg_total,confidence,
                kickoff,factors_json,logged_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (pid, pick["match"], pick["league"], pick.get("league_id", ""),
             pick["bet"], pick["bet_type"], pick["xg_total"], pick["confidence"],
             pick["kickoff_utc"], json.dumps(pick.get("factors", {})),
             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        )
        conn.commit()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  TIMEZONE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def store_datetime(dt: datetime) -> datetime:
    """Ensure datetime is UTC before storage."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def display_datetime(dt: datetime, user_tz_str: str = "UTC") -> str:
    """Convert UTC datetime to user-local timezone for display only."""
    try:
        user_tz = pytz.timezone(user_tz_str)
        local_dt = dt.astimezone(user_tz)
        return local_dt.strftime("%d %b · %H:%M %Z")
    except Exception:
        return dt.strftime("%d %b · %H:%M UTC")


def parse_utc(utc_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    except Exception:
        return None


def in_window(utc_str: str) -> bool:
    dt = parse_utc(utc_str)
    if not dt:
        return False
    n = now_utc()
    return n <= dt <= n + timedelta(hours=WINDOW_HOURS)


def minutes_to_kickoff(utc_str: str) -> int:
    dt = parse_utc(utc_str)
    if not dt:
        return 9999
    return max(0, int((dt - now_utc()).total_seconds() / 60))


def format_kickoff(utc_str: str) -> str:
    tz_str = st.session_state.get("user_timezone", "UTC")
    dt = parse_utc(utc_str)
    if not dt:
        return "—"
    return display_datetime(dt, tz_str)


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP HELPERS WITH RATE LIMITING AND RETRIES
# ══════════════════════════════════════════════════════════════════════════════
def safe_get(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 6,
    retries: int = 1,
    delay_between: float = 0.5,
) -> Optional[dict]:
    h = {**HTTP_HEADERS, **(headers or {})}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=h, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code == 429:
                time.sleep(min(60.0, delay_between * (2 ** attempt)))
            elif code in (401, 403):
                return None  # Auth failure — don't retry
            elif attempt < retries:
                time.sleep(delay_between)
        except Exception:
            if attempt < retries:
                time.sleep(delay_between)
    return None


def safe_get_text(url: str, timeout: int = 8) -> Optional[str]:
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


# ── Score parser ───────────────────────────────────────────────────────────────
def _parse_score(raw: Any) -> int:
    if raw is None:
        return 0
    if isinstance(raw, dict):
        raw = raw.get("value", raw.get("displayValue", 0))
    try:
        return int(float(str(raw)))
    except (ValueError, TypeError):
        return 0


# ══════════════════════════════════════════════════════════════════════════════
#  ESPN DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════
def fetch_scoreboard(league_id: str) -> list[dict]:
    result: list[dict] = []
    for delta in [0, 1]:
        date_str = (now_utc() + timedelta(days=delta)).strftime("%Y%m%d")
        key = f"sb_{league_id}_{date_str}"
        cached = cache_get(key, ttl=300)
        if cached is not None:
            result.extend(cached)
            continue
        t0 = time.time()
        data = safe_get(f"{ESPN_SOCCER}/{league_id}/scoreboard", params={"dates": date_str})
        latency = (time.time() - t0) * 1000
        if not data:
            log_pipeline(f"espn_{league_id}", "degraded", f"No scoreboard data for {date_str}", latency_ms=latency)
            continue
        events: list[dict] = []
        for ev in data.get("events", []):
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            status_type = comp.get("status", {}).get("type", {})
            events.append({
                "event_id":  ev.get("id", ""),
                "date":      ev.get("date", ""),
                "home_id":   str(home_c.get("team", {}).get("id", "")),
                "home_name": home_c.get("team", {}).get("displayName", ""),
                "away_id":   str(away_c.get("team", {}).get("id", "")),
                "away_name": away_c.get("team", {}).get("displayName", ""),
                "status":    status_type.get("name", ""),
                "completed": status_type.get("completed", False),
                "league_id": league_id,
            })
        cache_set(key, events)
        result.extend(events)
        if events:
            log_pipeline("espn_scoreboard", "ok", f"{league_id}: {len(events)} events", rows=len(events), latency_ms=latency)
    return result


def fetch_team_schedule_espn(league_id: str, team_id: str) -> list[dict]:
    date_tag = now_utc().strftime("%Y%m%d")
    key = f"sched_{league_id}_{team_id}_{date_tag}"
    cached = cache_get(key, ttl=3600)
    if cached is not None:
        return cached
    data = safe_get(f"{ESPN_SOCCER}/{league_id}/teams/{team_id}/schedule")
    if not data:
        return []
    games: list[dict] = []
    for ev in data.get("events", []):
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            continue
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        hs = _parse_score(home_c.get("score"))
        ags = _parse_score(away_c.get("score"))
        ev_date = ev.get("date", "")
        valid, _ = validate_match_result(hs, ags)
        if not valid:
            continue

        # ── Extract halftime scores from ESPN linescores ──────────────────────
        ht_home, ht_away = None, None
        try:
            home_ls = home_c.get("linescores", [])
            away_ls = away_c.get("linescores", [])
            if home_ls and away_ls:
                # period 1 = halftime
                ht_home = _parse_score(home_ls[0].get("value", None)) if home_ls else None
                ht_away = _parse_score(away_ls[0].get("value", None)) if away_ls else None
        except Exception:
            ht_home = ht_away = None

        game_rec = {
            "date":       ev_date,
            "home_name":  home_c.get("team", {}).get("displayName", ""),
            "away_name":  away_c.get("team", {}).get("displayName", ""),
            "home_score": hs,
            "away_score": ags,
            "total":      hs + ags,
            "ht_home":    ht_home,
            "ht_away":    ht_away,
            "ht_total":   (ht_home + ht_away) if (ht_home is not None and ht_away is not None) else None,
        }
        games.append(game_rec)
        # Opportunistically store in training DB
        store_match_result(league_id,
            home_c.get("team", {}).get("displayName", ""),
            away_c.get("team", {}).get("displayName", ""),
            hs, ags, ev_date)
    games.sort(key=lambda g: g["date"])
    games = games[-HISTORY_GAMES:]
    cache_set(key, games)
    return games


# ── TheSportsDB supplement ─────────────────────────────────────────────────────
def fetch_tsdb_team_last15(team_name: str) -> list[dict]:
    key = f"tsdb_{hashlib.md5(team_name.encode()).hexdigest()[:8]}"
    cached = cache_get(key, ttl=7200)
    if cached is not None:
        return cached
    sr = safe_get(f"{TSDB_BASE}/searchteams.php", params={"t": team_name}, timeout=6)
    if not sr or not sr.get("teams"):
        return []
    team_id = sr["teams"][0].get("idTeam", "")
    if not team_id:
        return []
    er = safe_get(f"{TSDB_BASE}/eventslast15.php", params={"id": team_id}, timeout=6)
    if not er or not er.get("results"):
        return []
    games: list[dict] = []
    for ev in er["results"]:
        try:
            hs = int(ev.get("intHomeScore", 0) or 0)
            ags = int(ev.get("intAwayScore", 0) or 0)
            home = ev.get("strHomeTeam", "")
            away = ev.get("strAwayTeam", "")
            ev_date = ev.get("dateEvent", "") + "T12:00:00Z"
            if not home or not away:
                continue
            valid, _ = validate_match_result(hs, ags)
            if not valid:
                continue
            games.append({"date": ev_date, "home_name": home, "away_name": away,
                          "home_score": hs, "away_score": ags, "total": hs + ags})
        except Exception:
            pass
    cache_set(key, games)
    return games


def fetch_team_schedule(league_id: str, team_id: str, team_name: str) -> list[dict]:
    espn_games = fetch_team_schedule_espn(league_id, team_id)
    if len(espn_games) >= MIN_GAMES:
        return espn_games
    tsdb_games = fetch_tsdb_team_last15(team_name)
    if not tsdb_games:
        return espn_games
    seen: set[str] = set()
    combined: list[dict] = []
    for g in espn_games + tsdb_games:
        k = f"{g['date'][:10]}_{g.get('home_name','')}_{g.get('away_name','')}"
        if k not in seen:
            seen.add(k)
            combined.append(g)
    combined.sort(key=lambda g: g.get("date", ""))
    return combined[-HISTORY_GAMES:]


# ── ClubElo ────────────────────────────────────────────────────────────────────
def fetch_clubelo(team_name: str) -> Optional[float]:
    key = f"elo_{hashlib.md5(team_name.lower().encode()).hexdigest()[:8]}"
    cached = cache_get(key, ttl=86400 * 7)
    if cached is not None:
        return cached.get("elo")
    text = safe_get_text(f"{CLUBELO_BASE}/{requests.utils.quote(team_name)}", timeout=5)
    if not text:
        return None
    try:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return None
        last = lines[-1].split(",")
        elo = float(last[4]) if len(last) > 4 else None
        if elo:
            cache_set(key, {"elo": elo})
        return elo
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def safe_mean(lst: list) -> float:
    return float(np.mean(lst)) if lst else 0.0


def team_stats(games: list[dict], team_name: str) -> Optional[dict]:
    completed = [
        g for g in games
        if g.get("home_score", -1) >= 0 and g.get("away_score", -1) >= 0
    ]
    if len(completed) < MIN_GAMES:
        return None

    home_games = [g for g in completed if g.get("home_name", "") == team_name]
    away_games = [g for g in completed if g.get("away_name", "") == team_name]

    def _split_stats(gl: list[dict], scored_key: str, conceded_key: str):
        if not gl:
            return None, None, None, None
        sc = [g[scored_key] for g in gl]
        co = [g[conceded_key] for g in gl]
        tot = [s + c for s, c in zip(sc, co)]
        n = len(gl)
        return (
            safe_mean(sc),
            safe_mean(co),
            sum(1 for t in tot if t > 0.5) / n,
            sum(1 for s, c in zip(sc, co) if s > 0 and c > 0) / n,
        )

    h_sc, h_co, h_over05, h_btts = _split_stats(home_games, "home_score", "away_score")
    a_sc, a_co, a_over05, a_btts = _split_stats(away_games, "away_score", "home_score")

    all_sc, all_co, all_tot = [], [], []
    home_sc_list, home_co_list = [], []
    away_sc_list, away_co_list = [], []

    for g in completed:
        is_home = g.get("home_name", "") == team_name
        sc = g["home_score"] if is_home else g["away_score"]
        co = g["away_score"] if is_home else g["home_score"]
        all_sc.append(sc); all_co.append(co); all_tot.append(sc + co)
        if is_home:
            home_sc_list.append(sc); home_co_list.append(co)
        else:
            away_sc_list.append(sc); away_co_list.append(co)

    n = len(completed)
    avg_sc = safe_mean(all_sc)
    avg_co = safe_mean(all_co)
    avg_tot = safe_mean(all_tot)

    over05_r = sum(1 for t in all_tot if t > 0.5) / n
    over15_r = sum(1 for t in all_tot if t > 1.5) / n
    over25_r = sum(1 for t in all_tot if t > 2.5) / n
    under15_r = sum(1 for t in all_tot if t <= 1.0) / n
    btts_r = sum(1 for s, c in zip(all_sc, all_co) if s > 0 and c > 0) / n
    cs_r = sum(1 for c in all_co if c == 0) / n
    wins_r = sum(1 for s, c in zip(all_sc, all_co) if s > c) / n

    # Venue-split under 1.5 and clean sheet rates
    home_under15 = (
        sum(1 for s, c in zip(home_sc_list, home_co_list) if (s + c) <= 1) / len(home_sc_list)
        if home_sc_list else under15_r
    )
    away_under15 = (
        sum(1 for s, c in zip(away_sc_list, away_co_list) if (s + c) <= 1) / len(away_sc_list)
        if away_sc_list else under15_r
    )
    home_cs = (
        sum(1 for c in home_co_list if c == 0) / len(home_co_list)
        if home_co_list else cs_r
    )
    away_cs = (
        sum(1 for c in away_co_list if c == 0) / len(away_co_list)
        if away_co_list else cs_r
    )

    # Form (last 5 vs rest)
    recent5 = all_tot[-5:] if n >= 5 else all_tot
    older = all_tot[:-5] if n > 5 else all_tot
    form_score = max(0.0, min(1.0, 0.5 + (safe_mean(recent5) - safe_mean(older)) / 4.0))
    last3_avg = safe_mean(all_tot[-3:]) if n >= 3 else avg_tot

    # Streaks
    def _streak(lst: list[bool]) -> int:
        s = 0
        for v in reversed(lst):
            if v:
                s += 1
            else:
                break
        return s

    over05_flags = [t > 0.5 for t in all_tot]
    over15_flags = [t > 1.5 for t in all_tot]
    over25_flags = [t > 2.5 for t in all_tot]
    btts_flags   = [s > 0 and c > 0 for s, c in zip(all_sc, all_co)]

    # ── Halftime stats (where available from ESPN linescores) ─────────────────
    ht_totals = [g.get("ht_total") for g in completed if g.get("ht_total") is not None]
    ht_n = len(ht_totals)
    if ht_n >= 3:
        ht_avg = safe_mean(ht_totals)
        ht_under_rate = sum(1 for t in ht_totals if t < 0.5) / ht_n   # 0 goals in HT
        ht_over_rate  = sum(1 for t in ht_totals if t >= 1.0) / ht_n  # 1+ goals in HT
        ht_over15_rate = sum(1 for t in ht_totals if t > 1.5) / ht_n  # 2+ goals in HT
    else:
        # Estimate from full-time averages: typically 35-40% of FT goals scored in HT
        ht_avg = avg_tot * 0.38
        ht_under_rate = max(0.1, min(0.9, 1.0 - (avg_tot * 0.38)))
        ht_over_rate  = max(0.1, min(0.9, avg_tot * 0.38)       )
        ht_over15_rate = max(0.05, min(0.7, avg_tot * 0.38 - 0.3))

    return {
        "n": n, "n_home": len(home_games), "n_away": len(away_games),
        "avg_scored": avg_sc, "avg_conceded": avg_co, "avg_total": avg_tot,
        "over05_rate": over05_r, "over15_rate": over15_r,
        "over25_rate": over25_r, "btts_rate": btts_r,
        "under15_rate": under15_r,
        "cs_rate": cs_r, "wins_rate": wins_r,
        # Venue splits (home at home)
        "home_avg_scored":   h_sc   if h_sc   is not None else avg_sc,
        "home_avg_conceded": h_co   if h_co   is not None else avg_co,
        "home_over05_rate":  h_over05 if h_over05 is not None else over05_r,
        "home_btts_rate":    h_btts if h_btts is not None else btts_r,
        "home_under15_rate": home_under15,
        "home_cs_rate":      home_cs,
        # Venue splits (away at away)
        "away_avg_scored":   a_sc   if a_sc   is not None else avg_sc,
        "away_avg_conceded": a_co   if a_co   is not None else avg_co,
        "away_over05_rate":  a_over05 if a_over05 is not None else over05_r,
        "away_btts_rate":    a_btts if a_btts is not None else btts_r,
        "away_under15_rate": away_under15,
        "away_cs_rate":      away_cs,
        # Form
        "form_score": form_score, "last3_avg": last3_avg,
        # Streaks
        "streak_over05": _streak(over05_flags),
        "streak_over15": _streak(over15_flags),
        "streak_over25": _streak(over25_flags),
        "streak_btts":   _streak(btts_flags),
        # Halftime stats (new in v5)
        "ht_avg_goals":   round(ht_avg, 3),
        "ht_under_rate":  round(ht_under_rate, 3),
        "ht_over_rate":   round(ht_over_rate, 3),
        "ht_over15_rate": round(ht_over15_rate, 3),
        "ht_n":           ht_n,
    }


def get_h2h_stats(
    home_sched: list[dict], away_sched: list[dict],
    home_name: str, away_name: str
) -> Optional[dict]:
    seen: set[str] = set()
    totals, home_wins, away_wins, bttss, under15s = [], [], [], [], []
    for g in home_sched + away_sched:
        gk = f"{g.get('date','')[:10]}_{g.get('home_name','')}_{g.get('away_name','')}"
        if gk in seen:
            continue
        seen.add(gk)
        names = {g.get("home_name", ""), g.get("away_name", "")}
        if {home_name, away_name} != names:
            continue
        hs, ags = g.get("home_score", 0), g.get("away_score", 0)
        t = hs + ags
        totals.append(t)
        home_wins.append(1 if hs > ags else 0)
        away_wins.append(1 if ags > hs else 0)
        bttss.append(1 if hs > 0 and ags > 0 else 0)
        under15s.append(1 if t <= 1 else 0)
    if len(totals) < 3:
        return None
    n = len(totals)
    return {
        "over05":    sum(1 for t in totals if t > 0.5) / n,
        "over15":    sum(1 for t in totals if t > 1.5) / n,
        "over25":    sum(1 for t in totals if t > 2.5) / n,
        "under15":   sum(under15s) / n,
        "btts":      sum(bttss) / n,
        "home_w":    sum(home_wins) / n,
        "away_w":    sum(away_wins) / n,
        "count":     n,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  POISSON MATHEMATICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def _pois_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    try:
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    except Exception:
        return 0.0


def poisson_under_line(lam_home: float, lam_away: float, line: float) -> float:
    """P(total goals <= floor(line))"""
    p = 0.0
    threshold = int(line)
    for h in range(threshold + 1):
        for a in range(threshold + 1 - h):
            p += _pois_pmf(h, max(0.01, lam_home)) * _pois_pmf(a, max(0.01, lam_away))
    return max(0.0, min(1.0, p))


def poisson_over_line(lam_home: float, lam_away: float, line: float) -> float:
    return max(0.0, 1.0 - poisson_under_line(lam_home, lam_away, line - 0.001))


def poisson_btts(lam_home: float, lam_away: float) -> float:
    p_home_0 = _pois_pmf(0, max(0.01, lam_home))
    p_away_0 = _pois_pmf(0, max(0.01, lam_away))
    return max(0.0, min(1.0, (1 - p_home_0) * (1 - p_away_0)))


def poisson_home_win(lam_home: float, lam_away: float) -> float:
    p = 0.0
    for h in range(13):
        for a in range(13):
            if h > a:
                p += _pois_pmf(h, max(0.01, lam_home)) * _pois_pmf(a, max(0.01, lam_away))
    return max(0.0, min(1.0, p))


def poisson_away_win(lam_home: float, lam_away: float) -> float:
    p = 0.0
    for h in range(13):
        for a in range(13):
            if a > h:
                p += _pois_pmf(h, max(0.01, lam_home)) * _pois_pmf(a, max(0.01, lam_away))
    return max(0.0, min(1.0, p))


def compute_xg(home_st: dict, away_st: dict) -> tuple[float, float]:
    xg_h = 0.55 * home_st["home_avg_scored"] + 0.45 * away_st["away_avg_conceded"]
    xg_a = 0.55 * away_st["away_avg_scored"] + 0.45 * home_st["home_avg_conceded"]
    return max(0.05, xg_h), max(0.05, xg_a)


# ══════════════════════════════════════════════════════════════════════════════
#  DIXON-COLES MODEL
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DixonColesResult:
    fitted: bool
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.1
    rho: float = -0.1
    warning: Optional[str] = None
    n_matches: int = 0
    sparse_teams: list[str] = field(default_factory=list)
    fit_date: Optional[str] = None


def _dc_tau(h: int, a: int, lam1: float, lam2: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor."""
    if h == 0 and a == 0:
        return 1.0 - lam1 * lam2 * rho
    if h == 1 and a == 0:
        return 1.0 + lam1 * rho
    if h == 0 and a == 1:
        return 1.0 + lam2 * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _dc_neg_loglik(
    params: np.ndarray,
    teams: list[str],
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
) -> float:
    n_teams = len(teams)
    attack  = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    home_adv = params[2*n_teams]
    rho      = params[2*n_teams + 1]

    total = 0.0
    for i in range(len(home_goals)):
        h_idx, a_idx = home_idx[i], away_idx[i]
        hg, ag = int(home_goals[i]), int(away_goals[i])
        lam1 = math.exp(attack[h_idx] - defense[a_idx] + home_adv)
        lam2 = math.exp(attack[a_idx] - defense[h_idx])
        lam1 = max(1e-6, lam1)
        lam2 = max(1e-6, lam2)
        tau = _dc_tau(hg, ag, lam1, lam2, rho)
        if tau <= 0:
            total += weights[i] * -20.0  # Penalise
            continue
        ll = (
            math.log(tau)
            + hg * math.log(lam1) - lam1
            + ag * math.log(lam2) - lam2
        )
        total -= weights[i] * ll

    # L2 regularisation on attack/defense to prevent sparse-team overfitting
    reg_strength = 0.01
    total += reg_strength * float(np.sum(attack ** 2))
    total += reg_strength * float(np.sum(defense ** 2))
    return total


def fit_dixon_coles(league_id: str) -> DixonColesResult:
    """Fit Dixon-Coles model for a league from accumulated match results."""
    try:
        conn = get_db()
        rows = conn.execute(
            """SELECT home_team, away_team, home_goals, away_goals, match_date
               FROM match_results
               WHERE league_id=? AND validated=1
               ORDER BY match_date""",
            (league_id,)
        ).fetchall()
    except Exception as e:
        return DixonColesResult(fitted=False, warning=f"DB error: {e}")

    n = len(rows)
    if n < DC_MIN_LEAGUE_MATCHES:
        return DixonColesResult(
            fitted=False,
            n_matches=n,
            warning=f"Only {n} matches (need {DC_MIN_LEAGUE_MATCHES}) — using pooled prior"
        )

    # Build arrays
    home_teams = [r[0] for r in rows]
    away_teams = [r[1] for r in rows]
    home_goals = np.array([int(r[2]) for r in rows])
    away_goals = np.array([int(r[3]) for r in rows])
    dates      = [r[4] for r in rows]

    teams = sorted(set(home_teams) | set(away_teams))
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    home_idx = np.array([team_idx[t] for t in home_teams])
    away_idx = np.array([team_idx[t] for t in away_teams])

    # Time decay weights (xi=0.0065 ≈ half-life ~107 days)
    today = now_utc()
    weights = np.ones(n)
    for i, d in enumerate(dates):
        try:
            dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
            days_ago = max(0, (today - dt).days)
            weights[i] = math.exp(-0.0065 * days_ago)
        except Exception:
            weights[i] = 0.5

    # Check sparse teams
    team_counts: dict[str, int] = {}
    for t in home_teams + away_teams:
        team_counts[t] = team_counts.get(t, 0) + 1
    sparse_teams = [t for t, c in team_counts.items() if c < DC_MIN_TEAM_MATCHES]

    # Initial parameters
    x0 = np.zeros(2 * n_teams + 2)
    x0[2 * n_teams] = 0.1    # home advantage
    x0[2 * n_teams + 1] = -0.1  # rho

    # Bounds: rho ∈ (-0.5, 0.5), home_adv ∈ (-0.5, 1.0)
    bounds = (
        [(-3.0, 3.0)] * n_teams      # attack
        + [(-3.0, 3.0)] * n_teams    # defense
        + [(-0.5, 1.0)]              # home_adv
        + [(-0.5, 0.5)]              # rho
    )

    try:
        result = minimize(
            _dc_neg_loglik,
            x0=x0,
            args=(teams, home_idx, away_idx, home_goals, away_goals, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )
    except Exception as e:
        return DixonColesResult(fitted=False, warning=f"Optimizer error: {e}", n_matches=n)

    if not result.success and result.fun > 1e9:
        return DixonColesResult(
            fitted=False,
            warning=f"Optimizer failed to converge: {result.message}",
            n_matches=n
        )

    params = result.x
    attack  = {teams[i]: float(params[i])           for i in range(n_teams)}
    defense = {teams[i]: float(params[n_teams + i]) for i in range(n_teams)}
    home_adv = float(params[2 * n_teams])
    rho      = float(params[2 * n_teams + 1])

    return DixonColesResult(
        fitted=True,
        attack=attack,
        defense=defense,
        home_adv=home_adv,
        rho=rho,
        n_matches=n,
        sparse_teams=sparse_teams,
        fit_date=today.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def dc_predict_under15(
    dc: DixonColesResult,
    home_team: str,
    away_team: str,
    fallback_xg_h: float,
    fallback_xg_a: float,
) -> dict:
    """Compute P(total goals <= 1) from fitted Dixon-Coles model."""
    if not dc.fitted:
        # Use Poisson fallback
        p = poisson_under_line(fallback_xg_h, fallback_xg_a, 1.0)
        return {
            "probability": p,
            "usable": True,
            "source": "poisson_fallback",
            "warning": dc.warning,
        }

    # Use league mean for unseen teams
    mean_attack  = float(np.mean(list(dc.attack.values()))) if dc.attack  else 0.0
    mean_defense = float(np.mean(list(dc.defense.values()))) if dc.defense else 0.0

    att_h  = dc.attack.get(home_team, mean_attack)
    def_a  = dc.defense.get(away_team, mean_defense)
    att_a  = dc.attack.get(away_team, mean_attack)
    def_h  = dc.defense.get(home_team, mean_defense)

    lam1 = max(1e-6, math.exp(att_h - def_a + dc.home_adv))
    lam2 = max(1e-6, math.exp(att_a - def_h))

    p_00 = _dc_tau(0, 0, lam1, lam2, dc.rho) * _pois_pmf(0, lam1) * _pois_pmf(0, lam2)
    p_10 = _dc_tau(1, 0, lam1, lam2, dc.rho) * _pois_pmf(1, lam1) * _pois_pmf(0, lam2)
    p_01 = _dc_tau(0, 1, lam1, lam2, dc.rho) * _pois_pmf(0, lam1) * _pois_pmf(1, lam2)
    prob = max(0.0, min(1.0, p_00 + p_10 + p_01))

    sparse = home_team in dc.sparse_teams or away_team in dc.sparse_teams
    return {
        "probability": prob,
        "usable": True,
        "source": "dixon_coles",
        "sparse_flag": sparse,
        "warning": f"Sparse data for {home_team if home_team in dc.sparse_teams else away_team}" if sparse else None,
    }


def get_dc_model(league_id: str) -> DixonColesResult:
    """Get or fit Dixon-Coles model for a league, cached in session_state."""
    cache_key = f"dc_{league_id}"
    dc_cache: dict[str, DixonColesResult] = st.session_state.get("dc_models", {})
    if cache_key in dc_cache:
        return dc_cache[cache_key]
    dc = fit_dixon_coles(league_id)
    dc_cache[cache_key] = dc
    st.session_state["dc_models"] = dc_cache
    return dc


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING FOR XGBoost
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_vector(
    home_st: dict,
    away_st: dict,
    h2h: Optional[dict],
    xg_h: float,
    xg_a: float,
) -> dict[str, float]:
    """Build a leakage-free feature vector for the Under 1.5 and HT models."""
    poisson_u15 = poisson_under_line(xg_h, xg_a, 1.0)
    # Halftime xG is approximately 38% of full-time xG
    xg_h_ht = xg_h * 0.38
    xg_a_ht = xg_a * 0.38
    poisson_ht_under = poisson_under_line(xg_h_ht, xg_a_ht, 0.5)  # P(HT total=0)
    poisson_ht_over  = poisson_over_line(xg_h_ht, xg_a_ht, 0.5)   # P(HT total>=1)

    h2h_under15 = h2h["under15"] if h2h else (
        (home_st["under15_rate"] + away_st["under15_rate"]) / 2.0
    )

    home_n_clipped = min(float(home_st["n"]), 38.0) / 38.0
    away_n_clipped = min(float(away_st["n"]), 38.0) / 38.0

    return {
        "home_avg_scored":     home_st["avg_scored"],
        "home_avg_conceded":   home_st["avg_conceded"],
        "away_avg_scored":     away_st["avg_scored"],
        "away_avg_conceded":   away_st["avg_conceded"],
        "home_under15_rate":   home_st["under15_rate"],
        "away_under15_rate":   away_st["under15_rate"],
        "home_under15_home":   home_st["home_under15_rate"],
        "away_under15_away":   away_st["away_under15_rate"],
        "home_cs_rate":        home_st["cs_rate"],
        "away_cs_rate":        away_st["cs_rate"],
        "home_cs_home":        home_st["home_cs_rate"],
        "away_cs_away":        away_st["away_cs_rate"],
        "h2h_under15":         h2h_under15,
        "poisson_under15":     poisson_u15,
        "home_form3":          home_st["last3_avg"],
        "away_form3":          away_st["last3_avg"],
        "home_n_clipped":      home_n_clipped,
        "away_n_clipped":      away_n_clipped,
        # Halftime features (v5 additions)
        "home_ht_under_rate":  home_st.get("ht_under_rate", 0.5),
        "away_ht_under_rate":  away_st.get("ht_under_rate", 0.5),
        "home_ht_over_rate":   home_st.get("ht_over_rate", 0.5),
        "away_ht_over_rate":   away_st.get("ht_over_rate", 0.5),
        "home_ht_avg_goals":   home_st.get("ht_avg_goals", 0.5),
        "away_ht_avg_goals":   away_st.get("ht_avg_goals", 0.5),
        "poisson_ht_under":    poisson_ht_under,
        "poisson_ht_over":     poisson_ht_over,
    }


def build_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Build XGBoost training dataset from accumulated match_results.
    Enforces temporal ordering — NO data leakage.
    """
    try:
        conn = get_db()
        rows = conn.execute(
            """SELECT league_id, home_team, away_team, home_goals, away_goals, match_date
               FROM match_results
               WHERE validated=1
               ORDER BY match_date ASC""",
        ).fetchall()
    except Exception:
        return pd.DataFrame(), pd.Series(dtype=int)

    if len(rows) < XGB_MIN_TRAIN:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Build team history dict incrementally (no leakage)
    team_history: dict[str, list[dict]] = {}

    feature_rows: list[dict] = []
    labels: list[int] = []

    for league_id, home_team, away_team, home_goals, away_goals, match_date in rows:
        # Get prior history (already seen matches only)
        home_games = team_history.get(home_team, [])
        away_games = team_history.get(away_team, [])

        if len(home_games) >= MIN_GAMES and len(away_games) >= MIN_GAMES:
            home_st = team_stats(home_games, home_team)
            away_st = team_stats(away_games, away_team)
            if home_st and away_st:
                xg_h, xg_a = compute_xg(home_st, away_st)
                h2h = get_h2h_stats(home_games, away_games, home_team, away_team)
                feats = build_feature_vector(home_st, away_st, h2h, xg_h, xg_a)
                feats["match_date"] = match_date
                feature_rows.append(feats)
                label = 1 if (int(home_goals) + int(away_goals)) <= 1 else 0
                labels.append(label)

        # NOW add this match to history (after using it as target)
        match_rec_home = {
            "date": match_date, "home_name": home_team, "away_name": away_team,
            "home_score": int(home_goals), "away_score": int(away_goals),
            "total": int(home_goals) + int(away_goals),
        }
        match_rec_away = dict(match_rec_home)
        team_history.setdefault(home_team, []).append(match_rec_home)
        team_history.setdefault(away_team, []).append(match_rec_away)

    if not feature_rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = pd.DataFrame(feature_rows)
    y  = pd.Series(labels, index=df.index, name="under15")
    return df, y


# ══════════════════════════════════════════════════════════════════════════════
#  XGBoost UNDER 1.5 CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
class Under15Model:
    """
    Production Under 1.5 Goals XGBoost classifier.
    - Temporal train/test split (no shuffling, no leakage)
    - scale_pos_weight for class imbalance
    - Platt scaling calibration
    - Lazy SHAP computation cached per match_id
    """

    def __init__(self) -> None:
        self.model: Optional[XGBClassifier] = None
        self.calibrator: Optional[LogisticRegression] = None
        self.trained: bool = False
        self.training_n: int = 0
        self.brier_val: float = 1.0
        self.calibration_n: int = 0
        self.fit_date: Optional[str] = None

    def train(self, df: pd.DataFrame, y: pd.Series) -> dict:
        n = len(df)
        if n < XGB_MIN_TRAIN:
            return {"trained": False, "reason": f"Need {XGB_MIN_TRAIN} samples, have {n}"}

        # Temporal split: 80% train, 20% calibration/test
        train_end = int(n * 0.80)
        calib_end = n

        df_sorted = df.sort_values("match_date")
        y_sorted  = y.loc[df_sorted.index]

        X_train = df_sorted.iloc[:train_end][FEATURE_COLS].values
        y_train = y_sorted.iloc[:train_end].values
        X_calib = df_sorted.iloc[train_end:calib_end][FEATURE_COLS].values
        y_calib = y_sorted.iloc[train_end:calib_end].values

        if len(X_calib) < XGB_MIN_CALIB:
            return {"trained": False, "reason": f"Calibration set too small: {len(X_calib)} < {XGB_MIN_CALIB}"}

        # Class imbalance
        n_under = int(y_train.sum())
        n_over  = len(y_train) - n_under
        spw = (n_over / n_under) if n_under > 0 else 2.0
        spw = max(0.5, min(5.0, spw))  # Guard against extreme ratios

        # Fixed well-tuned XGBoost params (no Optuna — too slow on cloud)
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
        )
        self.model.fit(X_train, y_train)

        # Platt scaling calibration
        calib_probs = self.model.predict_proba(X_calib)[:, 1]
        self.calibrator = LogisticRegression(C=1.0)
        self.calibrator.fit(calib_probs.reshape(-1, 1), y_calib)

        # Validation Brier score
        cal_probs = self.calibrator.predict_proba(calib_probs.reshape(-1, 1))[:, 1]
        self.brier_val    = float(brier_score_loss(y_calib, cal_probs))
        self.trained      = True
        self.training_n   = n
        self.calibration_n = len(y_calib)
        self.fit_date      = now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "trained": True,
            "brier_val": self.brier_val,
            "n": n,
            "calibration_n": self.calibration_n,
            "scale_pos_weight": round(spw, 3),
        }

    def predict(self, feat_dict: dict[str, float]) -> dict:
        if not self.trained or self.model is None:
            return {"probability": None, "usable": False, "reason": "Model not trained"}
        X = np.array([[feat_dict.get(c, 0.0) for c in FEATURE_COLS]])
        raw_prob = float(self.model.predict_proba(X)[0, 1])
        if self.calibrator:
            cal_prob = float(self.calibrator.predict_proba([[raw_prob]])[0, 1])
        else:
            cal_prob = raw_prob
        # Model disagreement from staged predictions (XGBoost 3.x compatible)
        try:
            import xgboost as xgb
            booster = self.model.get_booster()
            dmat = xgb.DMatrix(X)
            n_trees = self.model.n_estimators
            step = max(1, n_trees // 5)
            tree_preds = np.array([
                float(booster.predict(dmat, output_margin=True,
                                      iteration_range=(0, i))[0])
                for i in range(step, n_trees + 1, step)
            ])
            model_spread = float(np.std(1.0 / (1.0 + np.exp(-tree_preds)))) if len(tree_preds) > 1 else 0.0
        except Exception:
            model_spread = 0.0

        return {
            "probability":      cal_prob,
            "raw_probability":  raw_prob,
            "model_spread":     model_spread,
            "calibration_method": "platt",
            "calibration_n":    self.calibration_n,
            "brier_val":        self.brier_val,
            "training_n":       self.training_n,
            "usable":           True,
        }

    def get_shap_values(self, feat_dict: dict[str, float], match_id: str) -> Optional[dict]:
        """Compute SHAP values lazily, cached by match_id."""
        if not self.trained or self.model is None:
            return None
        cache_key = f"shap_{match_id}_{self.fit_date}"
        shap_cache = st.session_state.get("shap_cache", {})
        if cache_key in shap_cache:
            return shap_cache[cache_key]
        try:
            X = np.array([[feat_dict.get(c, 0.0) for c in FEATURE_COLS]])
            explainer = shap.TreeExplainer(self.model)
            # shap 0.51 compatible: handle both old ndarray and new Explanation API
            sv = explainer.shap_values(X)
            if hasattr(sv, "values"):
                # New shap Explanation object
                arr = sv.values
                shap_vals = (arr[0, :, 1] if arr.ndim == 3 else arr[0])
            elif isinstance(sv, list):
                # Binary clf returns list[neg_class, pos_class]
                shap_vals = sv[1][0] if len(sv) > 1 else sv[0][0]
            elif hasattr(sv, "ndim") and sv.ndim == 3:
                # (n_samples, n_features, n_classes)
                shap_vals = sv[0, :, 1]
            else:
                shap_vals = sv[0]
            result = {
                "feature_names": FEATURE_COLS,
                "shap_values":   shap_vals.tolist(),
                "base_value":    float(explainer.expected_value),
            }
            shap_cache[cache_key] = result
            st.session_state["shap_cache"] = shap_cache
            return result
        except Exception:
            return None


@st.cache_resource
def get_under15_model() -> Under15Model:
    """Singleton model object, persists across reruns."""
    return Under15Model()


def ensure_model_trained() -> dict:
    """Train XGBoost model if not yet trained and enough data exists."""
    model = get_under15_model()
    if model.trained:
        return {"ready": True, "training_n": model.training_n, "brier": model.brier_val}
    n = get_training_count()
    if n < XGB_MIN_TRAIN:
        return {"ready": False, "reason": f"Accumulating data: {n}/{XGB_MIN_TRAIN} matches stored"}
    df, y = build_training_data()
    if df.empty:
        return {"ready": False, "reason": "Insufficient feature data after temporal filtering"}
    result = model.train(df, y)
    if result.get("trained"):
        log_pipeline("xgb_model", "ok",
            f"Trained on {result['n']} matches, Brier={result['brier_val']:.4f}",
            rows=result["n"])
        st.session_state["model_initialized"] = True
        return {"ready": True, **result}
    return {"ready": False, "reason": result.get("reason", "Training failed")}


# ══════════════════════════════════════════════════════════════════════════════
#  ODDS API INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
def get_odds_api_key() -> Optional[str]:
    key = st.secrets.get("ODDS_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        key = os.environ.get("ODDS_API_KEY")
    # Fallback to hardcoded key provided by user
    return key or _HARDCODED_ODDS_KEY or None


def fetch_odds_under15(league_id: str) -> dict[str, dict]:
    """
    Fetch Under/Over totals odds from The Odds API.
    Returns dict: {event_key → {home, away, under_1_5_odds, over_1_5_odds, fetched_at}}

    Budget-conscious: fetches once per league per 15 minutes.
    Under 1.5 odds are in the 'totals' market with point=1.5.
    """
    api_key = get_odds_api_key()
    if not api_key:
        return {}

    # Map ESPN league_id → Odds API sport key (best effort)
    SPORT_MAP = {
        "eng.1": "soccer_epl",
        "esp.1": "soccer_spain_la_liga",
        "ger.1": "soccer_germany_bundesliga",
        "ita.1": "soccer_italy_serie_a",
        "fra.1": "soccer_france_ligue_one",
        "ned.1": "soccer_netherlands_eredivisie",
        "por.1": "soccer_portugal_primeira_liga",
        "usa.1": "soccer_usa_mls",
        "bra.1": "soccer_brazil_campeonato",
        "arg.1": "soccer_argentina_primera_division",
        "mex.1": "soccer_mexico_ligamx",
        "aus.1": "soccer_australia_aleague",
        "sco.1": "soccer_scotland_premiership",
        "tur.1": "soccer_turkey_super_league",
        "bel.1": "soccer_belgium_first_div",
        "sau.1": "soccer_saudi_arabians_premier_league",
        "jpn.1": "soccer_japan_j_league",
        "kor.1": "soccer_korea_kleague1",
        "chn.1": "soccer_china_superleague",
    }
    sport_key = SPORT_MAP.get(league_id)
    if not sport_key:
        return {}

    cache_key = f"odds_{league_id}_{now_utc().strftime('%Y%m%d%H')}_{now_utc().minute // 15}"
    cached = cache_get(cache_key, ttl=900)
    if cached is not None:
        log_pipeline("odds_api", "ok", f"Odds from cache for {league_id}", rows=len(cached))
        return cached

    t0 = time.time()
    data = safe_get(
        f"{ODDS_API_BASE}/sports/{sport_key}/odds",
        params={
            "apiKey": api_key,
            "regions": "uk,eu",
            "markets": "totals",
            "oddsFormat": "decimal",
            "commenceTimeFrom": now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commenceTimeTo": (now_utc() + timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        timeout=12,
    )
    latency = (time.time() - t0) * 1000

    if not data or not isinstance(data, list):
        log_pipeline("odds_api", "failed", f"No data for {league_id}", latency_ms=latency)
        return {}

    result: dict[str, dict] = {}
    fetched_at = now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")

    for event in data:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        bookmakers = event.get("bookmakers", [])

        best_under15: Optional[float] = None
        best_over15: Optional[float] = None
        all_odds: dict[str, float] = {}

        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") != "totals":
                    continue
                for outcome in market.get("outcomes", []):
                    try:
                        point = float(outcome.get("point", 0))
                        price = float(outcome.get("price", 0))
                        name  = outcome.get("name", "").lower()
                        if abs(point - 1.5) < 0.01:
                            if "under" in name:
                                all_odds["under_1_5"] = price
                                if best_under15 is None or price > best_under15:
                                    best_under15 = price
                            elif "over" in name:
                                all_odds["over_1_5"] = price
                                if best_over15 is None or price > best_over15:
                                    best_over15 = price
                    except Exception:
                        pass

        if best_under15 is not None:
            event_key = f"{home}_vs_{away}".replace(" ", "_")
            result[event_key] = {
                "home":           home,
                "away":           away,
                "commence_time":  commence,
                "under_1_5_odds": best_under15,
                "over_1_5_odds":  best_over15,
                "all_odds":       all_odds,
                "fetched_at":     fetched_at,
            }

    cache_set(cache_key, result)
    log_pipeline("odds_api", "ok", f"{league_id}: {len(result)} events with U1.5 odds",
                 rows=len(result), latency_ms=latency)

    # Track remaining quota from headers (if available via requests)
    return result


def match_odds(
    home_name: str, away_name: str, odds_data: dict[str, dict]
) -> Optional[dict]:
    """Fuzzy match ESPN team names to Odds API event."""
    if not odds_data:
        return None
    # Try exact match first
    key_exact = f"{home_name}_vs_{away_name}".replace(" ", "_")
    if key_exact in odds_data:
        return odds_data[key_exact]
    # Try partial matching
    home_lower = home_name.lower()
    away_lower = away_name.lower()
    for _, event in odds_data.items():
        oh = event.get("home", "").lower()
        oa = event.get("away", "").lower()
        if (home_lower[:6] in oh or oh[:6] in home_lower) and \
           (away_lower[:6] in oa or oa[:6] in away_lower):
            return event
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  ODDS MARGIN STRIPPING — POWER METHOD
# ══════════════════════════════════════════════════════════════════════════════
def strip_margin_power_method(all_odds: dict[str, float]) -> Optional[float]:
    """
    Strips bookmaker margin using the Power method.
    More accurate than basic normalization.
    Returns true implied probability for under_1_5.
    """
    if not all_odds or len(all_odds) < 2:
        return None
    raw_probs = {k: 1.0 / v for k, v in all_odds.items() if v > 1.0}
    if len(raw_probs) < 2:
        return None
    overround = sum(raw_probs.values())
    if overround <= 1.0:
        return None

    def equation(k: float) -> float:
        return sum(p ** k for p in raw_probs.values()) - 1.0

    try:
        k = brentq(equation, 0.3, 3.0, xtol=1e-6)
        true_probs = {outcome: p ** k for outcome, p in raw_probs.items()}
        return true_probs.get("under_1_5") or true_probs.get("under") or None
    except ValueError:
        # Fallback: simple normalization
        if "under_1_5" in raw_probs:
            return raw_probs["under_1_5"] / overround
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO KELLY CRITERION
# ══════════════════════════════════════════════════════════════════════════════
def compute_kelly(
    model_prob: float,
    implied_prob: float,
    odds: float,
    n_correlated: int = 1,
) -> dict:
    """
    Compute Kelly stake with correlation discount.
    n_correlated: number of same-league same-day bets.
    """
    edge = model_prob - implied_prob
    if edge <= 0:
        return {"kelly": 0.0, "edge": edge, "ev": 0.0, "recommended": 0.0}

    b = odds - 1.0
    q = 1.0 - model_prob
    if b <= 0:
        return {"kelly": 0.0, "edge": edge, "ev": 0.0, "recommended": 0.0}

    full_kelly = max(0.0, (b * model_prob - q) / b)
    ev = edge * b - (1 - model_prob)

    # Correlation discount: 1=100%, 2=70%, 3+=50%
    corr_factor = {1: 1.0, 2: 0.70}.get(n_correlated, 0.50)
    recommended = min(full_kelly * KELLY_FRACTION * corr_factor, KELLY_MAX_STAKE)

    return {
        "full_kelly":    round(full_kelly, 4),
        "recommended":   round(recommended, 4),
        "edge":          round(edge, 4),
        "ev":            round(ev, 4),
        "corr_factor":   corr_factor,
        "n_correlated":  n_correlated,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VALUE BET ENGINE — UNDER 1.5
# ══════════════════════════════════════════════════════════════════════════════
def scan_value_bets_under15() -> tuple[list[dict], dict]:
    """
    Scan all leagues for Under 1.5 value bets.
    Returns (value_bets, model_status).
    """
    model_status = ensure_model_trained()
    model = get_under15_model()
    value_bets: list[dict] = []
    candidates: list[dict] = []

    for league_id, league_name, flag in LEAGUES[:20]:  # Top 20 leagues for performance
        odds_data = fetch_odds_under15(league_id)
        events = fetch_scoreboard(league_id)
        window_events = [
            e for e in events
            if not e.get("completed", False) and in_window(e.get("date", ""))
        ]
        for ev in window_events:
            odds_match = match_odds(ev["home_name"], ev["away_name"], odds_data)
            if not odds_match:
                continue  # Under 1.5 market must exist — never fabricate
            under_odds = odds_match.get("under_1_5_odds")
            if not under_odds or under_odds <= 1.0:
                continue

            home_sched = fetch_team_schedule(league_id, ev["home_id"], ev["home_name"])
            away_sched = fetch_team_schedule(league_id, ev["away_id"], ev["away_name"])
            home_st = team_stats(home_sched, ev["home_name"])
            away_st = team_stats(away_sched, ev["away_name"])
            if not home_st or not away_st:
                continue

            xg_h, xg_a = compute_xg(home_st, away_st)
            h2h = get_h2h_stats(home_sched, away_sched, ev["home_name"], ev["away_name"])

            # Poisson baseline
            poisson_u15 = poisson_under_line(xg_h, xg_a, 1.0)

            # Dixon-Coles (if available)
            dc = get_dc_model(league_id)
            dc_result = dc_predict_under15(dc, ev["home_name"], ev["away_name"], xg_h, xg_a)
            dc_prob = dc_result["probability"]

            # XGBoost (if trained)
            feats = build_feature_vector(home_st, away_st, h2h, xg_h, xg_a)
            if model.trained:
                xgb_result = model.predict(feats)
                xgb_prob = xgb_result["probability"] if xgb_result["usable"] else None
                model_spread = xgb_result.get("model_spread", 0.0)
                calib_n = xgb_result.get("calibration_n", 0)
            else:
                xgb_prob = None
                model_spread = 0.0
                calib_n = 0

            # Ensemble: weight by availability
            if xgb_prob is not None:
                ensemble_prob = 0.50 * xgb_prob + 0.30 * dc_prob + 0.20 * poisson_u15
            else:
                ensemble_prob = 0.60 * dc_prob + 0.40 * poisson_u15

            # Odds margin stripping
            implied_prob = strip_margin_power_method(odds_match.get("all_odds", {}))
            if implied_prob is None:
                implied_prob = 1.0 / under_odds  # Simple fallback

            kelly = compute_kelly(ensemble_prob, implied_prob, under_odds)
            if kelly["edge"] <= 0:
                continue

            # Odds freshness check
            fetched_at = parse_utc(odds_match.get("fetched_at", ""))
            odds_age_min = 0.0
            if fetched_at:
                odds_age_min = (now_utc() - fetched_at).total_seconds() / 60.0

            hours_to_kickoff = minutes_to_kickoff(ev["date"]) / 60.0

            candidates.append({
                "match":           f"{ev['home_name']} vs {ev['away_name']}",
                "home":            ev["home_name"],
                "away":            ev["away_name"],
                "league":          f"{flag} {league_name}",
                "league_id":       league_id,
                "kickoff_utc":     ev["date"],
                "kickoff_display": format_kickoff(ev["date"]),
                "xg_home":         round(xg_h, 3),
                "xg_away":         round(xg_a, 3),
                "ensemble_prob":   round(ensemble_prob, 4),
                "xgb_prob":        round(xgb_prob, 4) if xgb_prob else None,
                "dc_prob":         round(dc_prob, 4),
                "poisson_prob":    round(poisson_u15, 4),
                "implied_prob":    round(implied_prob, 4),
                "edge":            kelly["edge"],
                "ev":              kelly["ev"],
                "full_kelly":      kelly["full_kelly"],
                "recommended_kelly": kelly["recommended"],
                "corr_factor":     kelly["corr_factor"],
                "under_odds":      under_odds,
                "odds_age_min":    round(odds_age_min, 1),
                "hours_to_kickoff": round(hours_to_kickoff, 2),
                "model_spread":    round(model_spread, 4),
                "calib_n":         calib_n,
                "dc_fitted":       dc.fitted,
                "dc_warning":      dc_result.get("warning"),
                "features":        feats,
                "h2h_count":       h2h["count"] if h2h else 0,
                "home_n":          home_st["n"],
                "away_n":          away_st["n"],
                "odds_source":     "the_odds_api",
            })

    # Apply correlation discount across same-league same-day
    league_date_counts: dict[str, int] = {}
    for c in candidates:
        ldk = f"{c['league_id']}_{c['kickoff_utc'][:10]}"
        league_date_counts[ldk] = league_date_counts.get(ldk, 0) + 1

    for c in candidates:
        ldk = f"{c['league_id']}_{c['kickoff_utc'][:10]}"
        n_corr = league_date_counts[ldk]
        if n_corr > 1:
            kelly2 = compute_kelly(c["ensemble_prob"], c["implied_prob"],
                                    c["under_odds"], n_corr)
            c["recommended_kelly"] = kelly2["recommended"]
            c["corr_factor"] = kelly2["corr_factor"]
            c["n_correlated"] = n_corr
        value_bets.append(c)

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets[:10], model_status


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-BET CONFIDENCE ENGINES (V3 adapted)
# ══════════════════════════════════════════════════════════════════════════════
def compute_over_confidence(
    home_st: dict, away_st: dict, line: float,
    h2h: Optional[dict], bet_type: str
) -> tuple[float, dict[str, float], str]:
    xg_h, xg_a = compute_xg(home_st, away_st)
    total_xg = xg_h + xg_a
    pois_p = poisson_over_line(xg_h, xg_a, line)

    hist_combined = (
        home_st.get("home_over05_rate" if line <= 0.5 else "over15_rate" if line <= 1.5 else "over25_rate",
                    home_st["over05_rate"]) * 0.5 +
        away_st.get("away_over05_rate" if line <= 0.5 else "over15_rate" if line <= 1.5 else "over25_rate",
                    away_st["over05_rate"]) * 0.5
    )
    xg_min = {0.5: 0.3, 1.5: 0.8, 2.5: 1.5}.get(line, 1.5)
    xg_max = {0.5: 2.0, 1.5: 3.5, 2.5: 5.0}.get(line, 5.0)
    xg_norm = max(0.0, min(1.0, (total_xg - xg_min) / (xg_max - xg_min)))
    btts_combined = (home_st["home_btts_rate"] + away_st["away_btts_rate"]) / 2
    form_combined = (home_st["form_score"] + away_st["form_score"]) / 2
    streak_key = {0.5: "streak_over05", 1.5: "streak_over15", 2.5: "streak_over25"}.get(line, "streak_over25")
    streak_val = min(1.0, (home_st.get(streak_key, 0) + away_st.get(streak_key, 0)) / 10.0)
    h2h_key = {0.5: "over05", 1.5: "over15", 2.5: "over25"}.get(line, "over25")
    h2h_val = h2h[h2h_key] if h2h else hist_combined

    factors = {
        "poisson_p": pois_p, "hist_rate": hist_combined, "xg_norm": xg_norm,
        "form": form_combined, "btts": btts_combined,
        "streak": streak_val, "h2h": h2h_val,
    }
    weights = get_weights(bet_type)
    confidence = sum(factors.get(k, 0.5) * w for k, w in weights.items()) * 100
    confidence = max(0.0, min(99.9, confidence))

    reasoning = (
        f"Poisson P(OVER {line}): {pois_p*100:.1f}% · "
        f"xG {xg_h:.2f}+{xg_a:.2f}={total_xg:.2f} · "
        f"Hist OVER {line}: {hist_combined*100:.0f}% · "
        f"BTTS: {btts_combined*100:.0f}%"
    )
    if home_st.get(streak_key, 0) >= 3:
        reasoning += f" · Home {home_st[streak_key]}-game streak 🔥"
    if away_st.get(streak_key, 0) >= 3:
        reasoning += f" · Away {away_st[streak_key]}-game streak 🔥"
    if h2h:
        reasoning += f" · H2H({h2h['count']}g) OVER {line}: {h2h[h2h_key]*100:.0f}%"
    return round(confidence, 1), factors, reasoning


def compute_btts_confidence(
    home_st: dict, away_st: dict, h2h: Optional[dict]
) -> tuple[float, dict[str, float], str]:
    xg_h, xg_a = compute_xg(home_st, away_st)
    pois_btts  = poisson_btts(xg_h, xg_a)
    hist_btts  = (home_st["home_btts_rate"] + away_st["away_btts_rate"]) / 2
    xg_balance = min(xg_h, xg_a) / max(xg_h, xg_a) if max(xg_h, xg_a) > 0 else 0.5
    form_comb  = (home_st["form_score"] + away_st["form_score"]) / 2
    h2h_btts   = h2h["btts"] if h2h else hist_btts

    factors = {
        "poisson_btts": pois_btts, "hist_btts": hist_btts,
        "xg_balance": xg_balance, "form": form_comb, "h2h": h2h_btts,
    }
    weights = get_weights("BTTS_YES")
    confidence = sum(factors.get(k, 0.5) * w for k, w in weights.items()) * 100
    confidence = max(0.0, min(99.9, confidence))

    reasoning = (
        f"Poisson BTTS: {pois_btts*100:.1f}% · Hist BTTS: {hist_btts*100:.0f}% · "
        f"xG balance: {xg_balance:.2f} · xG {xg_h:.2f} vs {xg_a:.2f}"
    )
    if h2h:
        reasoning += f" · H2H BTTS: {h2h['btts']*100:.0f}%"
    return round(confidence, 1), factors, reasoning


def compute_result_confidence(
    home_st: dict, away_st: dict,
    h2h: Optional[dict], side: str
) -> tuple[float, dict[str, float], str]:
    xg_h, xg_a = compute_xg(home_st, away_st)
    if side == "HOME":
        pois_p    = poisson_home_win(xg_h, xg_a)
        hist_rate = home_st["wins_rate"] * 0.6 + (1 - away_st["wins_rate"]) * 0.4
        form_diff = max(0.0, min(1.0, 0.5 + (home_st["form_score"] - away_st["form_score"]) / 2))
        xg_diff   = max(0.0, min(1.0, (xg_h - xg_a + 3) / 6))
        h2h_val   = h2h["home_w"] if h2h else hist_rate
        bt        = "HOME_WIN"
        factors   = {"poisson_hw": pois_p, "hist_hw": hist_rate, "form_diff": form_diff,
                     "xg_diff": xg_diff, "h2h": h2h_val}
    else:
        pois_p    = poisson_away_win(xg_h, xg_a)
        hist_rate = away_st["wins_rate"] * 0.6 + (1 - home_st["wins_rate"]) * 0.4
        form_diff = max(0.0, min(1.0, 0.5 + (away_st["form_score"] - home_st["form_score"]) / 2))
        xg_diff   = max(0.0, min(1.0, (xg_a - xg_h + 3) / 6))
        h2h_val   = h2h["away_w"] if h2h else hist_rate
        bt        = "AWAY_WIN"
        factors   = {"poisson_aw": pois_p, "hist_aw": hist_rate, "form_diff": form_diff,
                     "xg_diff": xg_diff, "h2h": h2h_val}

    weights    = get_weights(bt)
    confidence = sum(factors.get(k, 0.5) * w for k, w in weights.items()) * 100
    confidence = max(0.0, min(99.9, confidence))
    label      = "Home" if side == "HOME" else "Away"
    reasoning  = (
        f"Poisson {label} Win: {pois_p*100:.1f}% · "
        f"Hist Win Rate: {hist_rate*100:.0f}% · "
        f"xG {xg_h:.2f} vs {xg_a:.2f}"
    )
    if h2h:
        reasoning += f" · H2H {label} Win: {h2h_val*100:.0f}%"
    return round(confidence, 1), factors, reasoning


# ══════════════════════════════════════════════════════════════════════════════
#  ZEUS v5: FOCUSED SCANNER — FT Under 1.5 · HT Over/Under 1.5 ONLY
# ══════════════════════════════════════════════════════════════════════════════
def compute_ft_under15_confidence(
    home_st: dict, away_st: dict, h2h: Optional[dict],
    xg_h: float, xg_a: float
) -> tuple[float, dict[str, float], str]:
    """Compute confidence for Full-Time UNDER 1.5 Goals."""
    poisson_p = poisson_under_line(xg_h, xg_a, 1.0)
    hist_under15 = (home_st["under15_rate"] + away_st["under15_rate"]) / 2
    cs_rate = (home_st["cs_rate"] + away_st["cs_rate"]) / 2
    ht_under_rate = (home_st.get("ht_under_rate", 0.5) + away_st.get("ht_under_rate", 0.5)) / 2
    form = 1.0 - (home_st["form_score"] + away_st["form_score"]) / 2  # low scoring = inverted form
    h2h_val = h2h["under15"] if h2h else hist_under15

    factors = {
        "poisson_p":    poisson_p,
        "hist_under15": hist_under15,
        "cs_rate":      cs_rate,
        "ht_under_rate": ht_under_rate,
        "form":         max(0.0, min(1.0, form)),
        "h2h":          h2h_val,
    }
    weights = get_weights("FT_UNDER_15")
    confidence = sum(factors.get(k, 0.5) * w for k, w in weights.items()) * 100
    confidence = max(0.0, min(99.9, confidence))

    reasoning = (
        f"Poisson P(FT ≤1): {poisson_p*100:.1f}% · "
        f"xG {xg_h:.2f}+{xg_a:.2f}={xg_h+xg_a:.2f} · "
        f"Hist Under 1.5: {hist_under15*100:.0f}% · "
        f"Clean Sheet Rate: {cs_rate*100:.0f}%"
    )
    if home_st.get("ht_n", 0) >= 3:
        reasoning += f" · HT 0-goal rate: {ht_under_rate*100:.0f}%"
    if h2h:
        reasoning += f" · H2H({h2h['count']}g) Under 1.5: {h2h_val*100:.0f}%"
    return round(confidence, 1), factors, reasoning


def compute_ht_confidence(
    home_st: dict, away_st: dict, h2h: Optional[dict],
    xg_h: float, xg_a: float, bet_type: str
) -> tuple[float, dict[str, float], str]:
    """Compute confidence for HT UNDER 0.5 or HT OVER 0.5 Goals."""
    xg_h_ht = xg_h * 0.38
    xg_a_ht = xg_a * 0.38

    if bet_type == "HT_UNDER_15":
        # HT Under 0.5: P(no goal in first half)
        ht_poisson = poisson_under_line(xg_h_ht, xg_a_ht, 0.5)
        ht_hist = (home_st.get("ht_under_rate", 0.5) + away_st.get("ht_under_rate", 0.5)) / 2
        cs_rate = (home_st["cs_rate"] + away_st["cs_rate"]) / 2
        form = 1.0 - (home_st["form_score"] + away_st["form_score"]) / 2

        factors = {
            "ht_poisson":    ht_poisson,
            "ht_hist_under": ht_hist,
            "cs_rate":       cs_rate,
            "form":          max(0.0, min(1.0, form)),
            "h2h":           h2h["under15"] if h2h else ht_hist,
        }
        label = "HT UNDER 0.5"
        hist_pct = ht_hist * 100
    else:
        # HT Over 0.5: P(at least 1 goal in first half)
        ht_poisson = poisson_over_line(xg_h_ht, xg_a_ht, 0.5)
        ht_hist = (home_st.get("ht_over_rate", 0.5) + away_st.get("ht_over_rate", 0.5)) / 2
        form = (home_st["form_score"] + away_st["form_score"]) / 2
        over15_rate = (home_st.get("over15_rate", 0.5) + away_st.get("over15_rate", 0.5)) / 2

        factors = {
            "ht_poisson":   ht_poisson,
            "ht_hist_over": ht_hist,
            "form":         max(0.0, min(1.0, form)),
            "over15_rate":  over15_rate,
            "h2h":          h2h["over15"] if h2h else ht_hist,
        }
        label = "HT OVER 0.5"
        hist_pct = ht_hist * 100

    weights = get_weights(bet_type)
    confidence = sum(factors.get(k, 0.5) * w for k, w in weights.items()) * 100
    confidence = max(0.0, min(99.9, confidence))

    avg_ht = (home_st.get("ht_avg_goals", xg_h_ht) + away_st.get("ht_avg_goals", xg_a_ht)) / 2
    reasoning = (
        f"Poisson P({label}): {ht_poisson*100:.1f}% · "
        f"HT xG {xg_h_ht:.2f}+{xg_a_ht:.2f} · "
        f"Hist {label}: {hist_pct:.0f}% · "
        f"Avg HT goals: {avg_ht:.2f}"
    )
    ht_data_note = f"({home_st.get('ht_n',0)} HT records)" if home_st.get('ht_n', 0) > 0 else "(estimated from FT)"
    reasoning += f" {ht_data_note}"
    if h2h:
        reasoning += f" · H2H({h2h['count']}g)"
    return round(confidence, 1), factors, reasoning


@st.cache_data(ttl=300, show_spinner=False)
def scan_all_leagues() -> tuple[list[dict], int, int, int]:
    """Zeus v5: Scan ONLY for FT Under 1.5 and HT Over/Under 1.5 picks."""
    candidates: list[dict] = []
    leagues_hit = games_eval = data_pts = 0

    for league_id, league_name, flag in LEAGUES:
        events = fetch_scoreboard(league_id)
        if not events:
            continue
        window_games = [
            e for e in events
            if not e.get("completed", False) and in_window(e.get("date", ""))
        ]
        if not window_games:
            continue
        leagues_hit += 1

        for ev in window_games:
            home_sched = fetch_team_schedule(league_id, ev["home_id"], ev["home_name"])
            away_sched = fetch_team_schedule(league_id, ev["away_id"], ev["away_name"])
            data_pts  += len(home_sched) + len(away_sched)
            home_st    = team_stats(home_sched, ev["home_name"])
            away_st    = team_stats(away_sched, ev["away_name"])
            if home_st is None or away_st is None:
                continue
            games_eval += 1
            h2h = get_h2h_stats(home_sched, away_sched, ev["home_name"], ev["away_name"])
            xg_h, xg_a = compute_xg(home_st, away_st)
            total_xg   = round(xg_h + xg_a, 2)

            base = {
                "match":        f"{ev['home_name']} vs {ev['away_name']}",
                "home":         ev["home_name"],
                "away":         ev["away_name"],
                "league":       f"{flag} {league_name}",
                "league_id":    league_id,
                "kickoff_utc":  ev["date"],
                "kickoff_display": format_kickoff(ev["date"]),
                "mins_away":    minutes_to_kickoff(ev["date"]),
                "xg_total":     total_xg,
                "xg_home":      round(xg_h, 2),
                "xg_away":      round(xg_a, 2),
                "home_n":       home_st["n"],
                "away_n":       away_st["n"],
                "home_form":    home_st["form_score"],
                "away_form":    away_st["form_score"],
                "home_btts":    round(home_st.get("home_btts_rate", 0) * 100),
                "away_btts":    round(away_st.get("away_btts_rate", 0) * 100),
                "h2h_count":    h2h["count"] if h2h else 0,
                "home_ht_n":    home_st.get("ht_n", 0),
                "away_ht_n":    away_st.get("ht_n", 0),
                "home_ht_under": round(home_st.get("ht_under_rate", 0) * 100),
                "away_ht_under": round(away_st.get("ht_under_rate", 0) * 100),
                "home_ht_over":  round(home_st.get("ht_over_rate", 0) * 100),
                "away_ht_over":  round(away_st.get("ht_over_rate", 0) * 100),
            }

            # ── ONLY evaluate FT_UNDER_15, HT_UNDER_15, HT_OVER_15 ──────────
            for bet_type, bt_meta in BET_TYPES.items():
                gate = bt_meta["gate"]

                if bet_type == "FT_UNDER_15":
                    conf, factors, reasoning = compute_ft_under15_confidence(
                        home_st, away_st, h2h, xg_h, xg_a)
                    pois_p = poisson_under_line(xg_h, xg_a, 1.0)
                    extra = {
                        "poisson_p":    round(pois_p, 4),
                        "under15_hist": round((home_st["under15_rate"] + away_st["under15_rate"]) / 2 * 100),
                        "cs_avg":       round((home_st["cs_rate"] + away_st["cs_rate"]) / 2 * 100),
                    }
                elif bet_type in ("HT_UNDER_15", "HT_OVER_15"):
                    conf, factors, reasoning = compute_ht_confidence(
                        home_st, away_st, h2h, xg_h, xg_a, bet_type)
                    xg_h_ht, xg_a_ht = xg_h * 0.38, xg_a * 0.38
                    if bet_type == "HT_UNDER_15":
                        pois_p = poisson_under_line(xg_h_ht, xg_a_ht, 0.5)
                    else:
                        pois_p = poisson_over_line(xg_h_ht, xg_a_ht, 0.5)
                    extra = {
                        "poisson_p":   round(pois_p, 4),
                        "ht_hist_pct": round(
                            (home_st.get("ht_under_rate" if bet_type == "HT_UNDER_15" else "ht_over_rate", 0.5) +
                             away_st.get("ht_under_rate" if bet_type == "HT_UNDER_15" else "ht_over_rate", 0.5)) / 2 * 100
                        ),
                        "xg_ht": round(xg_h_ht + xg_a_ht, 3),
                    }
                else:
                    continue

                if conf < gate:
                    continue

                if conf >= 80:
                    tier, tier_label = "elite", "🔥 ELITE"
                elif conf >= 70:
                    tier, tier_label = "strong", "⚡ STRONG"
                else:
                    tier, tier_label = "result", "✅ CONFIDENT"

                candidates.append({
                    **base,
                    "bet_type":   bet_type,
                    "bet":        f"{bt_meta['emoji']} {bt_meta['label']}",
                    "confidence": conf,
                    "tier":       tier,
                    "tier_label": tier_label,
                    "reasoning":  reasoning,
                    "factors":    factors,
                    **extra,
                })

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    # Allow multiple bet types per match (FT Under + HT predictions are independent)
    top_picks: list[dict] = []
    seen_match_bet: set[str] = set()
    for c in candidates:
        key = f"{c['match']}_{c['bet_type']}"
        if key not in seen_match_bet:
            seen_match_bet.add(key)
            top_picks.append(c)
        if len(top_picks) >= TOP_N * 2:  # Show more picks since we have 3 focused bet types
            break

    for i, p in enumerate(top_picks, 1):
        p["rank"] = i
        save_pick(p)

    return top_picks, leagues_hit, games_eval, data_pts


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-GRADER AND ADAPTIVE LEARNER
# ══════════════════════════════════════════════════════════════════════════════
def grade_and_learn() -> int:
    conn = get_db()
    try:
        pending = conn.execute(
            "SELECT id, match, league_id, kickoff, bet_type, factors_json FROM picks_log WHERE result='pending'"
        ).fetchall()
    except Exception:
        return 0

    updated = 0
    for row_id, match, league_id, kickoff, bet_type, factors_json_str in pending:
        ko = parse_utc(kickoff)
        if not ko or (now_utc() - ko).total_seconds() < 6000:
            continue
        if not league_id:
            continue
        parts = match.split(" vs ")
        if len(parts) != 2:
            continue
        home_name, away_name = parts[0].strip(), parts[1].strip()
        date_str = ko.strftime("%Y%m%d")
        data = safe_get(f"{ESPN_SOCCER}/{league_id}/scoreboard", params={"dates": date_str})
        if not data:
            continue

        for ev in data.get("events", []):
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            names = {c.get("team", {}).get("displayName", "") for c in competitors}
            if home_name not in names and away_name not in names:
                continue
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            hs  = _parse_score(home_c.get("score"))
            ags = _parse_score(away_c.get("score"))
            tot = hs + ags

            valid, _ = validate_match_result(hs, ags)
            if not valid:
                break

            result_map = {
                "OVER_25":    "WON" if tot > 2.5 else "LOST",
                "OVER_05":    "WON" if tot > 0.5 else "LOST",
                "OVER_15":    "WON" if tot > 1.5 else "LOST",
                "BTTS_YES":   "WON" if hs > 0 and ags > 0 else "LOST",
                "HOME_WIN":   "WON" if hs > ags else "LOST",
                "AWAY_WIN":   "WON" if ags > hs else "LOST",
                # v5 focused bet types
                "FT_UNDER_15": "WON" if tot <= 1 else "LOST",
                "HT_UNDER_15": "PENDING",  # requires halftime score — graded separately
                "HT_OVER_15":  "PENDING",  # requires halftime score — graded separately
            }
            result = result_map.get(bet_type or "FT_UNDER_15", "WON" if tot <= 1 else "LOST")

            conn.execute(
                "UPDATE picks_log SET result=?,home_score=?,away_score=? WHERE id=?",
                (result, hs, ags, row_id)
            )
            updated += 1
            try:
                factors = json.loads(factors_json_str or "{}")
                if factors and bet_type:
                    update_weights(bet_type, factors, won=(result == "WON"))
            except Exception:
                pass
            break

    if updated:
        conn.commit()
    return updated


# ══════════════════════════════════════════════════════════════════════════════
#  HTML COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
def countdown_html(kickoff_utc: str, pick_id: str) -> str:
    return f"""
<div id="cd_{pick_id}" class="countdown">⏱ Calculating...</div>
<script>
(function(){{
  var target = new Date("{kickoff_utc}");
  var el = document.getElementById("cd_{pick_id}");
  function tick(){{
    var now = new Date(), diff = target - now;
    if(diff<=0){{ el.innerHTML="🔴 LIVE NOW"; el.style.color="#ff1744"; return; }}
    var h=Math.floor(diff/3600000), m=Math.floor((diff%3600000)/60000), s=Math.floor((diff%60000)/1000);
    var p=[]; if(h>0) p.push(h+"h"); p.push(("0"+m).slice(-2)+"m"); p.push(("0"+s).slice(-2)+"s");
    el.innerHTML="⏱ KICKOFF IN "+p.join(" ");
  }}
  tick(); setInterval(tick,1000);
}})();
</script>
"""


def _form_label(score: float) -> str:
    if score >= 0.65: return "🔥 HOT"
    if score <= 0.35: return "❄️ COLD"
    return "➡️ STABLE"


# ── Multi-bet card renderer ────────────────────────────────────────────────────
def render_pick_card(pick: dict) -> None:
    tier      = pick["tier"]
    conf      = pick["confidence"]
    bet_type  = pick["bet_type"]
    bt_meta   = BET_TYPES.get(bet_type, BET_TYPES["FT_UNDER_15"])
    pick_id   = hashlib.md5(f"{pick['match']}{bet_type}".encode()).hexdigest()[:6]
    bar_w     = min(99, int(conf))
    pois_pct  = pick.get("poisson_p", 0) * 100

    # CSS class for bet type
    bet_css_map = {"FT_UNDER_15": "bet-under15", "HT_UNDER_15": "bet-htunder", "HT_OVER_15": "bet-htover"}
    bet_css = bet_css_map.get(bet_type, "bet-under15")

    h2h_html = (
        f'<span class="pill pill-h2h">H2H({pick["h2h_count"]}g)</span>'
        if pick.get("h2h_count", 0) > 0 else ""
    )

    conn = get_db()
    upd_row = conn.execute(
        "SELECT SUM(updates) FROM model_weights WHERE bet_type=?", (bet_type,)
    ).fetchone()
    total_upd = int(upd_row[0] or 0) if upd_row else 0
    learn_html = (
        f'<span class="pill pill-learn">🧠 LEARNT {total_upd}</span>'
        if total_upd > 0 else ""
    )

    # HT-specific data cells
    if bet_type == "FT_UNDER_15":
        stat1_val  = f"{pick.get('under15_hist', 0):.0f}%"
        stat1_lbl  = "Hist U1.5"
        stat2_val  = f"{pick.get('cs_avg', 0):.0f}%"
        stat2_lbl  = "Clean Sheet"
        stat3_val  = f"{pick['xg_total']:.2f}"
        stat3_lbl  = "xG Total"
    elif bet_type == "HT_UNDER_15":
        stat1_val  = f"{pick.get('home_ht_under', 0):.0f}%"
        stat1_lbl  = "Home HT 0g"
        stat2_val  = f"{pick.get('away_ht_under', 0):.0f}%"
        stat2_lbl  = "Away HT 0g"
        stat3_val  = f"{pick.get('xg_ht', pick['xg_total']*0.38):.2f}"
        stat3_lbl  = "HT xG"
    else:  # HT_OVER_15
        stat1_val  = f"{pick.get('home_ht_over', 0):.0f}%"
        stat1_lbl  = "Home HT 1g+"
        stat2_val  = f"{pick.get('away_ht_over', 0):.0f}%"
        stat2_lbl  = "Away HT 1g+"
        stat3_val  = f"{pick.get('xg_ht', pick['xg_total']*0.38):.2f}"
        stat3_lbl  = "HT xG"

    ht_data_pill = ""
    ht_n = pick.get("home_ht_n", 0) + pick.get("away_ht_n", 0)
    if ht_n > 0:
        ht_data_pill = f'<span class="pill pill-calib">📊 {ht_n} HT records</span>'
    else:
        ht_data_pill = f'<span class="pill pill-warn">⚠️ HT estimated</span>'

    card_html = f"""
<div class="pick-card {tier}">
  <div class="rank-badge">#{pick.get('rank', '')}</div>
  <div class="card-league">{pick['league']}</div>
  <div class="card-teams">{pick['home']} <span class="card-vs">vs</span> {pick['away']}</div>
  <div class="card-bet {bet_css}">{pick['bet']}</div>
  <div class="conf-row">
    <span class="conf-pct {tier}">{conf:.1f}%</span>
    <span class="tier-chip {tier}">{pick['tier_label']}</span>
  </div>
  <div class="conf-track"><div class="conf-fill {tier}" style="width:{bar_w}%;"></div></div>
  <div class="ai-grid">
    <div class="ai-factor">
      <span class="ai-factor-val cyan">{pois_pct:.1f}%</span>
      <div class="ai-factor-lbl">Poisson P</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val gold">{stat1_val}</span>
      <div class="ai-factor-lbl">{stat1_lbl}</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val teal">{stat2_val}</span>
      <div class="ai-factor-lbl">{stat2_lbl}</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val">{stat3_val}</span>
      <div class="ai-factor-lbl">{stat3_lbl}</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val">{_form_label(pick['home_form'])}</span>
      <div class="ai-factor-lbl">Home Form</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val purple">{pick['home_n']}+{pick['away_n']}</span>
      <div class="ai-factor-lbl">Games Data</div>
    </div>
  </div>
  <div class="pills-row">
    <span class="pill pill-time">{pick['kickoff_display']}</span>
    <span class="pill pill-xg">FT xG: {pick['xg_home']:.2f}+{pick['xg_away']:.2f}</span>
    {h2h_html}{learn_html}{ht_data_pill}
  </div>
  <div class="card-reason">{pick['reasoning']}</div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)
    try:
        st.html(countdown_html(pick["kickoff_utc"], pick_id))
    except Exception:
        try:
            import streamlit.components.v1 as components
            components.html(countdown_html(pick["kickoff_utc"], pick_id), height=28)
        except Exception:
            pass


# ── Value bet card renderer ────────────────────────────────────────────────────
def render_value_card(bet: dict, rank: int) -> None:
    match_id = hashlib.md5(f"{bet['match']}{bet['kickoff_utc']}".encode()).hexdigest()[:8]
    edge_pct = bet["edge"] * 100
    ev_pct   = bet["ev"] * 100
    bar_w    = min(99, max(1, int(edge_pct * 5)))

    model_components = []
    if bet.get("xgb_prob") is not None:
        model_components.append(f"XGB {bet['xgb_prob']*100:.1f}%")
    model_components.append(f"DC {bet['dc_prob']*100:.1f}%")
    model_components.append(f"Pois {bet['poisson_prob']*100:.1f}%")
    model_str = " · ".join(model_components)

    odds_warn_html = ""
    if bet["odds_age_min"] > 60:
        odds_warn_html = f'<span class="pill pill-warn">⚠️ Odds {bet["odds_age_min"]:.0f}min old</span>'

    corr_html = ""
    if bet.get("n_correlated", 1) > 1:
        corr_html = f'<span class="pill pill-warn">Corr discount ×{bet["corr_factor"]:.0%}</span>'

    dc_warn_html = ""
    if bet.get("dc_warning"):
        dc_warn_html = f'<span class="pill pill-warn">⚠️ DC: {bet["dc_warning"]}</span>'

    spread_label = f"±{bet['model_spread']*100:.1f}%" if bet['model_spread'] > 0 else "N/A"
    calib_label  = f"Calib n={bet['calib_n']}" if bet['calib_n'] > 0 else "Pois/DC"

    card_html = f"""
<div class="pick-card value">
  <div class="rank-badge">#{rank}</div>
  <div class="card-league">{bet['league']}</div>
  <div class="card-teams">{bet['home']} <span class="card-vs">vs</span> {bet['away']}</div>
  <div class="card-bet bet-under15">🎰 UNDER 1.5 GOALS</div>
  <div class="conf-row">
    <span class="conf-pct value">+{edge_pct:.1f}% EDGE</span>
    <span class="tier-chip value">VALUE BET</span>
  </div>
  <div class="conf-track"><div class="conf-fill value" style="width:{bar_w}%;"></div></div>
  <div class="ai-grid">
    <div class="ai-factor">
      <span class="ai-factor-val teal">{bet['ensemble_prob']*100:.1f}%</span>
      <div class="ai-factor-lbl">Ensemble P</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val gold">{bet['under_odds']:.2f}</span>
      <div class="ai-factor-lbl">Best Odds</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val">{bet['implied_prob']*100:.1f}%</span>
      <div class="ai-factor-lbl">Implied P*</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val cyan">{ev_pct:+.2f}%</span>
      <div class="ai-factor-lbl">Exp Value</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val gold">{bet['recommended_kelly']*100:.1f}%</span>
      <div class="ai-factor-lbl">Kelly Stake</div>
    </div>
    <div class="ai-factor">
      <span class="ai-factor-val purple">{bet['home_n']}+{bet['away_n']}</span>
      <div class="ai-factor-lbl">Games Data</div>
    </div>
  </div>
  <div class="pills-row">
    <span class="pill pill-time">{bet['kickoff_display']}</span>
    <span class="pill pill-odds">{bet['under_odds']:.2f} u1.5</span>
    <span class="pill pill-edge">+{edge_pct:.1f}% edge</span>
    <span class="pill pill-kelly">Kelly: {bet['recommended_kelly']*100:.1f}%</span>
    <span class="pill pill-calib">{calib_label}</span>
    {odds_warn_html}{corr_html}{dc_warn_html}
  </div>
  <div class="card-reason">{model_str} → Ensemble {bet['ensemble_prob']*100:.1f}% vs Implied {bet['implied_prob']*100:.1f}% · H2H: {bet.get('h2h_count',0)} meetings</div>
  <div class="card-disclaimer">* Implied probability stripped of margin via Power method. Model spread ≈ {spread_label} (model disagreement — not a CI). Always compare odds yourself before betting.</div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)
    try:
        st.html(countdown_html(bet["kickoff_utc"], f"v_{match_id}"))
    except Exception:
        pass


# ── Pipeline health banner ─────────────────────────────────────────────────────
def render_health_banner(sources: list[str]) -> None:
    health = st.session_state.get("pipeline_health", {})
    if not health:
        return
    relevant = {k: v for k, v in health.items() if any(s in k for s in sources)}
    if not relevant:
        return
    has_failed = any(v.get("status") == "failed" for v in relevant.values())
    has_degraded = any(v.get("status") == "degraded" for v in relevant.values())
    if has_failed:
        cls = "health-fail"
        icon = "🔴"
    elif has_degraded:
        cls = "health-warn"
        icon = "🟡"
    else:
        cls = "health-ok"
        icon = "🟢"
    msgs = [f"{v.get('message','')}" for v in list(relevant.values())[:2]]
    st.markdown(
        f'<div class="{cls}">{icon} {" · ".join(msgs)}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS (honest, sample-size gated)
# ══════════════════════════════════════════════════════════════════════════════
def compute_performance_metrics(conn: sqlite3.Connection) -> dict:
    try:
        rows = conn.execute(
            """SELECT result, confidence, xg_total FROM picks_log
               WHERE result IN ('WON','LOST')"""
        ).fetchall()
    except Exception:
        return {"available": False, "reason": "DB query failed"}

    n = len(rows)
    if n == 0:
        return {"available": False, "reason": "No graded picks yet"}

    won   = sum(1 for r in rows if r[0] == "WON")
    lost  = n - won
    acc   = won / n

    # Wilson CI on accuracy
    z = 1.96
    center = (acc + z*z / (2*n)) / (1 + z*z/n)
    margin = (z * math.sqrt(acc*(1-acc)/n + z*z/(4*n*n))) / (1 + z*z/n)
    ci_lo, ci_hi = max(0.0, center - margin), min(1.0, center + margin)

    result = {
        "available": True,
        "n": n,
        "won": won,
        "lost": lost,
        "accuracy": acc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "roi_available": False,
        "roi_sample_warning": None,
    }

    if n >= MIN_ROI_SAMPLE:
        result["roi_available"] = True
    else:
        result["roi_sample_warning"] = f"ROI hidden — need {MIN_ROI_SAMPLE} graded picks, have {n}"

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TAB RENDERERS
# ══════════════════════════════════════════════════════════════════════════════
def render_tab_picks(count: int, newly_graded: int) -> None:
    tz_str = st.session_state.get("user_timezone", "UTC")
    now_display = display_datetime(now_utc(), tz_str)
    st.caption(
        f"🕐 {now_display} &nbsp;·&nbsp; Scanning games in next {WINDOW_HOURS}h "
        f"&nbsp;·&nbsp; Auto-refresh 60s &nbsp;·&nbsp; Scan #{count or '—'}"
    )
    render_health_banner(["espn"])

    if newly_graded:
        st.toast(f"🧠 Zeus learned from {newly_graded} graded pick(s)!", icon="⚡")

    with st.spinner(""):
        st.markdown(
            '<div class="scan-line">⚡ ZEUS v5 SCANNING 75+ LEAGUES — FT UNDER 1.5 · HT OVER/UNDER 1.5 ⚡</div>',
            unsafe_allow_html=True
        )
        picks, leagues_hit, games_eval, data_pts = scan_all_leagues()

    elite_cnt    = sum(1 for p in picks if p["tier"] == "elite")
    ft_under_cnt = sum(1 for p in picks if p.get("bet_type") == "FT_UNDER_15")
    ht_under_cnt = sum(1 for p in picks if p.get("bet_type") == "HT_UNDER_15")
    ht_over_cnt  = sum(1 for p in picks if p.get("bet_type") == "HT_OVER_15")

    st.markdown(f"""
<div class="metrics-row">
  <div class="metric-box"><span class="metric-val">{len(picks)}</span><div class="metric-lbl">Picks Today</div></div>
  <div class="metric-box"><span class="metric-val gold">{elite_cnt}</span><div class="metric-lbl">🔥 Elite</div></div>
  <div class="metric-box"><span class="metric-val teal">{ft_under_cnt}</span><div class="metric-lbl">🔒 FT U1.5</div></div>
  <div class="metric-box"><span class="metric-val cyan">{ht_under_cnt}</span><div class="metric-lbl">🕐 HT U0.5</div></div>
  <div class="metric-box"><span class="metric-val gold">{ht_over_cnt}</span><div class="metric-lbl">⚡ HT O0.5</div></div>
  <div class="metric-box"><span class="metric-val">{leagues_hit}</span><div class="metric-lbl">Leagues Hit</div></div>
  <div class="metric-box"><span class="metric-val">{games_eval}</span><div class="metric-lbl">Games Eval</div></div>
  <div class="metric-box"><span class="metric-val cyan">{data_pts:,}</span><div class="metric-lbl">Data Points</div></div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    if not picks:
        st.markdown("""
<div class="no-picks">
  <span class="no-picks-icon">⏳</span>
  No FT Under 1.5 or HT Over/Under 1.5 picks meet Zeus's thresholds in the next 6 hours.<br>
  Scanning 75+ leagues continuously — check back as fixtures enter the window.
</div>
""", unsafe_allow_html=True)
    else:
        cols = st.columns(3)
        cols[0].markdown('<span style="font-family:Barlow Condensed;color:#ffb300;font-size:.85rem;">🔥 ELITE — Exceptional multi-model edge</span>', unsafe_allow_html=True)
        cols[1].markdown('<span style="font-family:Barlow Condensed;color:#1de9b6;font-size:.85rem;">🔒 FT UNDER 1.5 — Full-time low-scoring game</span>', unsafe_allow_html=True)
        cols[2].markdown('<span style="font-family:Barlow Condensed;color:#00e5ff;font-size:.85rem;">🕐 HT U0.5 · ⚡ HT O0.5 — First half totals</span>', unsafe_allow_html=True)
        st.markdown("---")
        for pick in picks:
            render_pick_card(pick)


def render_tab_value() -> None:
    st.subheader("🎰 Under 1.5 Goals Value Engine")
    render_health_banner(["odds_api", "xgb_model", "dc_"])

    n_training = get_training_count()
    model = get_under15_model()

    # Model status card
    with st.expander("📊 Model Status", expanded=not model.trained):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training Data", f"{n_training}/{XGB_MIN_TRAIN}", delta="needed" if n_training < XGB_MIN_TRAIN else "✓")
        c2.metric("XGB Model", "✅ Trained" if model.trained else "⏳ Pending",
                  delta=f"Brier {model.brier_val:.4f}" if model.trained else f"Need {XGB_MIN_TRAIN - n_training} more")
        c3.metric("Calibration", f"Platt n={model.calibration_n}" if model.trained else "—")
        c4.metric("Fit Date", model.fit_date[:10] if model.fit_date else "—")

        if n_training < XGB_MIN_TRAIN:
            st.progress(n_training / XGB_MIN_TRAIN)
            st.info(
                f"⏳ Accumulating training data from ESPN match history: **{n_training}/{XGB_MIN_TRAIN}** matches. "
                f"Data collected automatically as the app scans fixtures. "
                f"Come back in a few hours once sufficient data is gathered."
            )
        elif not model.trained:
            if st.button("🔄 Train Model Now", key="train_btn"):
                with st.spinner("Training XGBoost under 1.5 model..."):
                    result = ensure_model_trained()
                if result.get("ready"):
                    st.success(f"✅ Model trained! n={result.get('n')}, Brier={result.get('brier_val', 0):.4f}")
                    st.rerun()
                else:
                    st.error(result.get("reason", "Training failed"))

        if not get_odds_api_key():
            st.warning(
                "⚠️ **The Odds API key not configured.** Value bets require real odds data.\n\n"
                "Add `ODDS_API_KEY = 'your_key'` to your `.streamlit/secrets.toml` file "
                "or set the `ODDS_API_KEY` environment variable.\n\n"
                "Free tier: 500 requests/month. Get your key at https://the-odds-api.com/"
            )

    if not model.trained:
        st.info("Value bet engine will activate once the XGBoost model is trained (requires sufficient match data).")
        return

    if not get_odds_api_key():
        st.info("Connect The Odds API to enable value bet identification. Model probabilities shown below for reference.")
        # Show model probabilities without odds
        with st.spinner("Computing model probabilities..."):
            try:
                picks, leagues_hit, _, _ = scan_all_leagues()
            except Exception as e:
                st.error(f"Scan failed: {e}")
                return
        if not picks:
            st.info("No upcoming matches in prediction window.")
            return
        for pick in picks[:5]:
            feats = build_feature_vector(
                {"avg_scored":0.5,"avg_conceded":0.5,"under15_rate":0.3,
                 "home_under15_rate":0.3,"away_under15_rate":0.3,
                 "cs_rate":0.2,"home_cs_rate":0.2,"away_cs_rate":0.2,
                 "home_avg_scored":0.5,"home_avg_conceded":0.5,
                 "away_avg_scored":0.5,"away_avg_conceded":0.5,
                 "form_score":0.5,"last3_avg":1.5,"n":10,"n_home":5,"n_away":5,
                 "home_btts_rate":0.5,"away_btts_rate":0.5,"wins_rate":0.5},
                {"avg_scored":0.5,"avg_conceded":0.5,"under15_rate":0.3,
                 "home_under15_rate":0.3,"away_under15_rate":0.3,
                 "cs_rate":0.2,"home_cs_rate":0.2,"away_cs_rate":0.2,
                 "home_avg_scored":0.5,"home_avg_conceded":0.5,
                 "away_avg_scored":0.5,"away_avg_conceded":0.5,
                 "form_score":0.5,"last3_avg":1.5,"n":10,"n_home":5,"n_away":5,
                 "home_btts_rate":0.5,"away_btts_rate":0.5,"wins_rate":0.5},
                None, 0.7, 0.7
            )
        return

    # Full value bet scan
    with st.spinner("🔍 Scanning for Under 1.5 value bets..."):
        try:
            value_bets, model_status = scan_value_bets_under15()
        except Exception as e:
            st.error(f"Value scan error: {e}")
            return

    if not value_bets:
        st.markdown("""
<div class="no-picks">
  <span class="no-picks-icon">🎰</span>
  No Under 1.5 value bets found in current scan window.<br>
  Value bets appear when model probability exceeds market implied probability.<br>
  Scanning continuously — check back as odds update.
</div>
""", unsafe_allow_html=True)
        return

    st.markdown(f"""
<div class="metrics-row">
  <div class="metric-box"><span class="metric-val teal">{len(value_bets)}</span><div class="metric-lbl">Value Bets</div></div>
  <div class="metric-box"><span class="metric-val gold">{max(v['edge']*100 for v in value_bets):.1f}%</span><div class="metric-lbl">Best Edge</div></div>
  <div class="metric-box"><span class="metric-val">{max(v['under_odds'] for v in value_bets):.2f}</span><div class="metric-lbl">Best Odds</div></div>
  <div class="metric-box"><span class="metric-val cyan">{model.training_n}</span><div class="metric-lbl">Model Trained On</div></div>
  <div class="metric-box"><span class="metric-val">{model.brier_val:.4f}</span><div class="metric-lbl">Brier Score</div></div>
</div>
""", unsafe_allow_html=True)
    st.caption("⚠️ Kelly stakes are fractional (25%) with portfolio correlation discount. Max 5% of bankroll per bet. Always verify odds independently.")
    st.markdown("---")

    for i, bet in enumerate(value_bets, 1):
        render_value_card(bet, i)
        # SHAP breakdown (lazy)
        match_id = hashlib.md5(f"{bet['match']}{bet['kickoff_utc']}".encode()).hexdigest()[:8]
        with st.expander(f"📊 Model breakdown — {bet['match']}", expanded=False):
            if st.button("Load SHAP explanation", key=f"shap_{match_id}"):
                with st.spinner("Computing feature contributions..."):
                    shap_data = model.get_shap_values(bet["features"], match_id)
                if shap_data:
                    shap_df = pd.DataFrame({
                        "Feature":    shap_data["feature_names"],
                        "SHAP Value": shap_data["shap_values"],
                    }).sort_values("SHAP Value", key=abs, ascending=False)
                    st.bar_chart(shap_df.set_index("Feature")["SHAP Value"].head(10))
                    st.caption(f"Base value: {shap_data['base_value']:.4f} — positive = increases P(under 1.5), negative = decreases it")
                else:
                    st.info("SHAP computation unavailable for this match.")


def render_tab_results(newly_graded: int) -> None:
    st.subheader("🏆 Zeus Pick Results — Auto-Graded & Self-Learning")
    if newly_graded:
        st.success(f"✅ {newly_graded} pick(s) graded this refresh and learned from.")

    try:
        conn = get_db()
        perf = compute_performance_metrics(conn)

        if perf.get("available"):
            n, won, lost = perf["n"], perf["won"], perf["lost"]
            acc    = perf["accuracy"]
            ci_lo  = perf["ci_lo"]
            ci_hi  = perf["ci_hi"]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("✅ Won",        won)
            c2.metric("❌ Lost",       lost)
            c3.metric("Total Graded",  n)
            c4.metric("Accuracy",      f"{acc*100:.1f}%",
                      delta=f"95% CI: [{ci_lo*100:.0f}%–{ci_hi*100:.0f}%]")
            if perf.get("roi_sample_warning"):
                c5.metric("ROI", "—",
                          delta=perf["roi_sample_warning"], delta_color="off")
            else:
                c5.metric("ROI", "See details")

            st.caption(f"Wilson 95% CI on accuracy: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] — meaningful at n={n}")
        else:
            st.info(perf.get("reason", "No picks graded yet."))

        # Bet type breakdown
        rows = conn.execute(
            "SELECT bet_type, result FROM picks_log WHERE result IN ('WON','LOST')"
        ).fetchall()
        if rows:
            st.markdown("**Win Rate by Bet Type**")
            bt_data: dict[str, dict] = {}
            for bet_type, result in rows:
                if bet_type not in bt_data:
                    bt_data[bet_type] = {"won": 0, "total": 0}
                bt_data[bet_type]["total"] += 1
                if result == "WON":
                    bt_data[bet_type]["won"] += 1
            bt_rows = []
            for bt, counts in bt_data.items():
                label = BET_TYPES.get(bt, {}).get("label", bt)
                emoji = BET_TYPES.get(bt, {}).get("emoji", "")
                wr = counts["won"] / counts["total"] * 100 if counts["total"] else 0
                bt_rows.append({"Bet Type": f"{emoji} {label}", "Won": counts["won"],
                                "Total": counts["total"], "Win Rate %": round(wr, 1)})
            st.dataframe(
                pd.DataFrame(bt_rows),
                column_config={"Win Rate %": st.column_config.ProgressColumn("Win Rate %", min_value=0, max_value=100)},
                hide_index=True,
                use_container_width=True,
            )

        st.divider()
        all_rows = conn.execute(
            """SELECT match, league, bet, bet_type, xg_total, confidence,
                      kickoff, result, home_score, away_score, logged_at
               FROM picks_log ORDER BY logged_at DESC LIMIT 200"""
        ).fetchall()

        if not all_rows:
            st.info("No picks logged yet — visit 🎯 Top Picks to generate predictions.")
            return

        df = pd.DataFrame(all_rows, columns=[
            "Match", "League", "Bet", "Bet Type", "xG", "Conf%",
            "Kickoff UTC", "Result", "Home", "Away", "Logged At"
        ])
        df["Conf%"] = df["Conf%"].apply(lambda x: f"{x:.1f}%")
        df["xG"]    = df["xG"].apply(lambda x: f"{x:.2f}")

        tab_won, tab_lost, tab_pending = st.tabs(["✅ Won", "❌ Lost", "⏳ Pending"])
        with tab_won:
            won_df = df[df["Result"] == "WON"]
            if won_df.empty:
                st.info("No winning picks graded yet.")
            else:
                for _, r in won_df.iterrows():
                    score = f" | {int(r['Home'])}-{int(r['Away'])}" if int(r.get("Home", -1)) >= 0 else ""
                    st.markdown(
                        f"⚽ **{r['Match']}** · {r['League']} · **{r['Bet']}** · "
                        f"xG: {r['xG']} · Conf: **{r['Conf%']}**{score} · "
                        f"<span style='color:#39ff14;font-weight:700;'>WON ✅</span>",
                        unsafe_allow_html=True
                    )
                    st.divider()

        with tab_lost:
            lost_df = df[df["Result"] == "LOST"]
            if lost_df.empty:
                st.info("No missed picks yet.")
            else:
                for _, r in lost_df.iterrows():
                    score = f" | {int(r['Home'])}-{int(r['Away'])}" if int(r.get("Home", -1)) >= 0 else ""
                    st.markdown(
                        f"⚽ **{r['Match']}** · {r['League']} · **{r['Bet']}** · "
                        f"xG: {r['xG']} · Conf: **{r['Conf%']}**{score} · "
                        f"<span style='color:#ff1744;font-weight:700;'>MISSED ❌</span>",
                        unsafe_allow_html=True
                    )
                    st.divider()

        with tab_pending:
            pend_df = df[df["Result"] == "pending"]
            if pend_df.empty:
                st.info("No pending picks.")
            else:
                st.dataframe(
                    pend_df[["Match", "League", "Bet", "Conf%", "Kickoff UTC"]],
                    hide_index=True,
                    use_container_width=True,
                )

    except Exception as e:
        st.error(f"Results unavailable: {e}")


def render_tab_brain() -> None:
    st.subheader("🧠 Zeus Adaptive Intelligence — Model & Weight Tracker")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Adaptive Weights")
        st.markdown(
            "Zeus adjusts prediction factor weights after every graded pick. "
            "Winning picks reinforce high-contributing factors. "
            "Weights are normalized to sum to 1.0 after each update."
        )
        try:
            conn = get_db()
            rows = conn.execute(
                "SELECT bet_type, factor, weight, wins, losses, updates FROM model_weights ORDER BY bet_type, weight DESC"
            ).fetchall()
            if rows:
                df_w = pd.DataFrame(rows, columns=["Bet Type", "Factor", "Weight", "Wins", "Losses", "Updates"])
                for bt in BET_TYPES.keys():
                    bdf = df_w[df_w["Bet Type"] == bt].copy()
                    if bdf.empty:
                        continue
                    tot_upd = bdf["Updates"].astype(int).sum()
                    tot_w   = bdf["Wins"].astype(int).sum()
                    tot_l   = bdf["Losses"].astype(int).sum()
                    bdf["Weight%"] = bdf["Weight"].apply(lambda x: round(float(x)*100, 1))
                    with st.expander(
                        f"{BET_TYPES[bt]['emoji']} {BET_TYPES[bt]['label']} — "
                        f"{tot_upd} updates · {tot_w}W / {tot_l}L"
                    ):
                        st.dataframe(
                            bdf[["Factor", "Weight%", "Wins", "Losses", "Updates"]],
                            column_config={"Weight%": st.column_config.ProgressColumn("Weight%", min_value=0, max_value=100)},
                            hide_index=True,
                            use_container_width=True,
                        )
            else:
                st.info("Weights initialize on first scan.")
        except Exception as e:
            st.error(f"Weight data unavailable: {e}")

    with c2:
        st.markdown("### Under 1.5 XGBoost Model")
        model = get_under15_model()
        n_training = get_training_count()
        if model.trained:
            st.success(f"✅ Model active | n={model.training_n} | Brier={model.brier_val:.4f}")
            st.metric("Calibration samples", model.calibration_n)
            st.metric("Fit date", model.fit_date[:10] if model.fit_date else "—")
            st.markdown(f"""
**Model configuration:**
- Features: {len(FEATURE_COLS)} (all leakage-free)
- Temporal split: 80% train / 20% calibration
- Class imbalance: `scale_pos_weight` auto-computed
- Calibration: Platt scaling (logistic regression)
- Feature leakage prevention: temporal ordering enforced
- No future data ever used for training
""")
            # Retrain option
            if st.button("🔄 Force Retrain Model", key="retrain_brain"):
                model.trained = False
                model.model = None
                with st.spinner("Retraining..."):
                    result = ensure_model_trained()
                if result.get("ready"):
                    st.success(f"✅ Retrained! Brier={result.get('brier_val', 0):.4f}")
                    st.rerun()
                else:
                    st.error(result.get("reason", "Failed"))
        else:
            st.info(f"Model not yet trained. Data: {n_training}/{XGB_MIN_TRAIN}")
            st.progress(min(1.0, n_training / XGB_MIN_TRAIN))

        st.markdown("### Dixon-Coles Status")
        dc_models = st.session_state.get("dc_models", {})
        if dc_models:
            dc_rows = []
            for key, dc in list(dc_models.items())[:10]:
                league_id = key.replace("dc_", "")
                dc_rows.append({
                    "League": league_id,
                    "Fitted": "✅" if dc.fitted else "❌",
                    "Matches": dc.n_matches,
                    "Warning": dc.warning or "—",
                })
            st.dataframe(pd.DataFrame(dc_rows), hide_index=True, use_container_width=True)
        else:
            st.info("DC models will appear after value bet scans.")

        st.markdown("### Learning Parameters")
        st.markdown(f"""
- Learning rate: `{LEARNING_RATE}` per graded pick
- Signal: `+1.0` for wins, `-0.5` for losses (asymmetric)
- Weights normalized to sum=1.0 after each update
- DC min league matches: `{DC_MIN_LEAGUE_MATCHES}`
- XGB min training samples: `{XGB_MIN_TRAIN}`
- Kelly fraction: `{KELLY_FRACTION}` (quarter-Kelly)
- Kelly max stake: `{KELLY_MAX_STAKE*100:.0f}%` of bankroll
- ROI shown only at n≥{MIN_ROI_SAMPLE} (no vanity metrics)
""")


def render_tab_pipeline() -> None:
    st.subheader("🔧 Pipeline Health Dashboard")
    health = st.session_state.get("pipeline_health", {})

    if not health:
        st.info("No pipeline events recorded yet — health updates on first scan.")
    else:
        rows = []
        for source, info in sorted(health.items()):
            ts = info.get("timestamp")
            age = f"{(now_utc() - ts).total_seconds()/60:.0f}m ago" if ts else "—"
            rows.append({
                "Source":  source,
                "Status":  info.get("status", "—"),
                "Rows":    info.get("rows", 0),
                "Latency": f"{info.get('latency_ms', 0):.0f}ms",
                "Age":     age,
                "Message": info.get("message", "")[:80],
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            column_config={
                "Status": st.column_config.TextColumn("Status"),
                "Rows": st.column_config.NumberColumn("Rows"),
            },
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.markdown("### Recent Pipeline Log")
    try:
        conn = get_db()
        log_rows = conn.execute(
            "SELECT source, status, message, rows, latency_ms, logged_at FROM pipeline_log ORDER BY id DESC LIMIT 50"
        ).fetchall()
        if log_rows:
            log_df = pd.DataFrame(log_rows, columns=["Source", "Status", "Message", "Rows", "Latency ms", "Logged At"])
            st.dataframe(log_df, hide_index=True, use_container_width=True)
        else:
            st.info("Pipeline log is empty.")
    except Exception as e:
        st.error(f"Pipeline log unavailable: {e}")

    st.divider()
    st.markdown("### Data Sources — Zeus v5 Assessment")
    st.markdown("""
| Source | Type | Status | Notes |
|--------|------|--------|-------|
| ESPN Soccer API | Free, no key | 🟢 Always available | Primary fixture + schedule + halftime scores |
| TheSportsDB | Free, key=3 | 🟢 Always available | Supplementary when ESPN thin |
| ClubElo API | Free, no key | 🟡 Best-effort | Weekly Elo ratings, not critical |
| The Odds API | Key built-in | ✅ Active | FT Under 1.5 value bets — HT odds via totals |
| Dixon-Coles | Computed | 🟡 Needs data | Requires ≥150 matches per league |
| XGBoost Model | Computed | 🟡 Needs data | Requires ≥300 labeled matches |

**v5 Focus — What Zeus v5 predicts ONLY:**
- 🔒 **Full-Time Under 1.5 Goals** — using Poisson, Dixon-Coles, XGBoost + historical rates
- 🕐 **Half-Time Under 0.5 Goals** — using ESPN halftime linescores + Poisson HT xG
- ⚡ **Half-Time Over 0.5 Goals** — attacking teams likely to score in first half

**What Zeus v5 does NOT predict:**
- Over 2.5, BTTS, Home Win, Away Win, or any other market — these are removed
""")

    st.markdown("### Training Data Status")
    n = get_training_count()
    col1, col2, col3 = st.columns(3)
    col1.metric("Stored Match Results", n)
    col2.metric("Required for XGB", XGB_MIN_TRAIN)
    col3.metric("Required for DC", DC_MIN_LEAGUE_MATCHES)
    if n > 0:
        st.progress(min(1.0, n / XGB_MIN_TRAIN), text=f"{n}/{XGB_MIN_TRAIN} matches for XGBoost")

    # League-level training data
    try:
        conn = get_db()
        league_rows = conn.execute(
            "SELECT league_id, COUNT(*) as cnt FROM match_results WHERE validated=1 GROUP BY league_id ORDER BY cnt DESC"
        ).fetchall()
        if league_rows:
            league_df = pd.DataFrame(league_rows, columns=["League ID", "Match Count"])
            league_df["DC Ready"] = league_df["Match Count"].apply(
                lambda x: "✅" if x >= DC_MIN_LEAGUE_MATCHES else f"⏳ ({x}/{DC_MIN_LEAGUE_MATCHES})"
            )
            st.dataframe(league_df, hide_index=True, use_container_width=True)
    except Exception:
        pass


def render_tab_settings() -> None:
    st.subheader("⚙️ Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Display Settings")
        selected_tz = st.selectbox(
            "Display Timezone",
            options=TIMEZONE_OPTIONS,
            index=TIMEZONE_OPTIONS.index(st.session_state.get("user_timezone", "UTC")),
            key="tz_select",
            help="All kickoff times displayed in this timezone. Data stored in UTC.",
        )
        if selected_tz != st.session_state.get("user_timezone"):
            st.session_state["user_timezone"] = selected_tz
            st.toast(f"Timezone updated to {selected_tz}", icon="🕐")

        st.markdown(f"**Current UTC time:** {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        st.markdown(f"**Local time ({selected_tz}):** {display_datetime(now_utc(), selected_tz)}")

    with col2:
        st.markdown("### API Keys")
        odds_key = get_odds_api_key()
        if odds_key:
            st.success(f"✅ The Odds API key configured (ends in ...{odds_key[-4:]})")
            if odds_key == _HARDCODED_ODDS_KEY:
                st.info("🔑 Using built-in API key. To override, set ODDS_API_KEY in secrets.toml or env.")
        else:
            st.warning("⚠️ The Odds API key not set — value bet engine inactive")
            st.markdown("""
Add your key to `.streamlit/secrets.toml`:
```toml
ODDS_API_KEY = "your_key_here"
```
Or set as environment variable `ODDS_API_KEY`.
Get a free key at [the-odds-api.com](https://the-odds-api.com/)
""")

        st.markdown("### Maintenance")
        if st.button("🗑️ Clear API Cache", key="clear_cache"):
            try:
                conn = get_db()
                conn.execute("DELETE FROM api_cache")
                conn.commit()
                st.cache_data.clear()
                st.toast("Cache cleared!", icon="🗑️")
            except Exception as e:
                st.error(f"Failed: {e}")

        if st.button("🔄 Reset DC Models", key="reset_dc"):
            st.session_state["dc_models"] = {}
            st.toast("DC models reset — will refit on next scan", icon="🔄")

        if st.button("🔄 Reset XGB Model", key="reset_xgb"):
            model = get_under15_model()
            model.trained = False
            model.model = None
            st.session_state["model_initialized"] = False
            st.toast("XGBoost model reset — will retrain when data sufficient", icon="🔄")

    st.divider()
    st.markdown("### System Information")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
**Zeus v{APP_VERSION} — FT U1.5 · HT O/U Specialist**
- Platform: Streamlit Community Cloud
- RAM budget: 1GB hard ceiling
- Filesystem: Ephemeral (/tmp/)
- DB: SQLite at {DB_PATH}
- Refresh: 60s auto-refresh
- Bet types: FT Under 1.5 · HT Under 0.5 · HT Over 0.5
""")
    with col4:
        st.markdown(f"""
**Model Configuration**
- XGB min train: {XGB_MIN_TRAIN} matches
- DC min league: {DC_MIN_LEAGUE_MATCHES} matches
- Kelly fraction: {KELLY_FRACTION} (quarter-Kelly)
- Max stake: {KELLY_MAX_STAKE*100:.0f}% of bankroll
- Correlation discount: 2 bets=70%, 3+=50%
- ROI gate: {MIN_ROI_SAMPLE} predictions minimum
""")

    st.divider()
    with st.expander("ℹ️ Leagues Monitored (75+)"):
        league_data = [
            {"Flag": f, "League": ln, "ID": lid}
            for lid, ln, f in LEAGUES
        ]
        st.dataframe(pd.DataFrame(league_data), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Deferred startup: let Streamlit's health check pass before any network calls ──
    # On the very first run, we just render the hero + a loading spinner, then
    # immediately rerun so the health check succeeds.  All heavyweight work
    # (scan_all_leagues, grade_and_learn) only runs from the second render onward.
    if "zeus_ready" not in st.session_state:
        st.session_state["zeus_ready"] = False

    # Auto-refresh (60 s)
    count = 0
    if _AUTOREFRESH_OK:
        count = st_autorefresh(interval=60_000, key="zeus_v5_autorefresh")
    st.session_state["refresh_count"] = count

    # Hero banner — always shown immediately
    st.markdown("""
<div class="zeus-hero">
  <span class="zeus-logo">⚡ ZEUS</span>
  <div class="zeus-tagline">Neural Football Intelligence · FT Under 1.5 · HT Over/Under 1.5 Specialist</div>
  <div class="zeus-version">
    v5.0 · Dixon-Coles · XGBoost · Portfolio Kelly · Pipeline Health · 75+ Leagues
  </div>
  <div class="zeus-bar"></div>
</div>
""", unsafe_allow_html=True)

    # First render: show a warm-up screen and trigger immediate rerun
    if not st.session_state["zeus_ready"]:
        st.markdown("""
<div style="text-align:center;padding:3rem 0;font-family:'Barlow Condensed',sans-serif;">
  <div style="font-size:3rem;">⚡</div>
  <div style="font-size:1.4rem;color:#1de9b6;margin:.5rem 0;">Zeus v5 is initialising…</div>
  <div style="color:#888;font-size:.95rem;">Connecting to ESPN · TheSportsDB · The Odds API</div>
</div>
""", unsafe_allow_html=True)
        st.session_state["zeus_ready"] = True
        st.rerun()
        return  # safety — rerun() raises, but keeps linters happy

    # All subsequent renders: full UI
    newly_graded = grade_and_learn()

    tab_picks, tab_value, tab_results, tab_brain, tab_pipeline, tab_settings = st.tabs([
        "🎯 FT U1.5 · HT O/U",
        "🎰 Value Engine",
        "🏆 Results",
        "🧠 AI Brain",
        "🔧 Pipeline",
        "⚙️ Settings",
    ])

    with tab_picks:
        render_tab_picks(count, newly_graded)

    with tab_value:
        render_tab_value()

    with tab_results:
        render_tab_results(newly_graded)

    with tab_brain:
        render_tab_brain()

    with tab_pipeline:
        render_tab_pipeline()

    with tab_settings:
        render_tab_settings()


if __name__ == "__main__":
    main()
