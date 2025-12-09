import streamlit as st
from io import BytesIO
from api_client import APIClient
import os
from PIL import Image
import base64

# --- App title ---
st.set_page_config(page_title="Digital Closet", layout="wide")

# --- Global theming (single CSS block) ---
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@200;300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

/* Apply Nunito font to body and all elements */
body, * {
  font-family: 'Nunito', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Prevent horizontal scrolling at root level */
html, body {
  overflow-x: hidden !important;
  max-width: 100vw !important;
}

/* Main background + text */
.stApp { 
  background:#FFFCF5; 
  color:#2E2E2E;
  overflow-x: hidden !important;
}

/* Header (top bar) - cream to match body */
header[data-testid="stHeader"]{
  background:#FFFCF5 !important;
  border-bottom:none !important; height:50px;
}

/* Sidebar - same as main body for sleek look */
section[data-testid="stSidebar"]{
  background:#FFFCF5; color:#6B4C98; text-align:center;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] p{
  color:#6B4C98 !important; text-align:center !important;
}

/* Buttons (main) — mauve bg, CREAM text */
.stButton > button:first-child{
  background:#B392AC !important; border:none !important; border-radius:8px !important;
  width:100%; padding:0.6em 1em; margin:6px 0;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
.stButton > button:first-child *{ color:#FFFCF5 !important; fill:#FFFCF5 !important; }
.stButton > button:hover{ background:#735D78 !important; }
.stButton > button:hover *{ color:#FFFCF5 !important; fill:#FFFCF5 !important; }

/* Sidebar buttons - Clear with purple text by default, dark fill when active */
section[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  border-radius: 8px !important;
  color: #6B4C98 !important;
  width: 100% !important;
  padding: 0.6em 1em !important;
  margin: 6px 0 !important;
  text-align: left !important;
  transition: background-color 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button * {
  color: #6B4C98 !important;
  fill: #6B4C98 !important;
}
/* Active page button - dark fill */
section[data-testid="stSidebar"] .stButton > button.active-page {
  background: #715d78 !important;
  background-color: #715d78 !important;
  border: 1px solid #D1B3C4 !important;
}
section[data-testid="stSidebar"] .stButton > button.active-page * {
  color: #FFFFFF !important;
  fill: #FFFFFF !important;
}
/* Active/clicked state - dark fill */
section[data-testid="stSidebar"] .stButton > button:active,
section[data-testid="stSidebar"] .stButton > button:focus {
  background: #715d78 !important;
  background-color: #715d78 !important;
  border: 1px solid #D1B3C4 !important;
}
section[data-testid="stSidebar"] .stButton > button:active *,
section[data-testid="stSidebar"] .stButton > button:focus * {
  color: #FFFFFF !important;
  fill: #FFFFFF !important;
}
/* Hover state - light fill */
section[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(247, 233, 253, 0.5) !important;
  background-color: rgba(247, 233, 253, 0.5) !important;
}

/* Make center container transparent so cream shows through */
.main .block-container{ 
  background:transparent !important;
  max-width: 100% !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  box-sizing: border-box !important;
}

/* Main content area - prevent overflow */
.main {
  overflow-x: hidden !important;
  max-width: 100% !important;
}

/* Ensure columns align properly at all zoom levels */
.stColumns, [data-testid="stHorizontalBlock"] {
  gap: 0.5rem !important;
  flex-wrap: nowrap !important;
  max-width: 100% !important;
}
[data-testid="column"] {
  min-width: 0 !important;
  overflow: hidden !important;
}

/* Headings in main area */
h1,h2,h3,h4,h5{ color:#6B4C98; }
/* My Wardrobe title - slightly bigger */
div[data-testid="stSubheader"] h2 {
  font-size: 1.5rem !important;
}

/* Horizontal logo directly above My Wardrobe - no spacing */
.main .block-container > div:has(img[src*="horizontal_logo"]) {
  margin: 0 !important;
  padding: 0 !important;
  margin-bottom: 0 !important;
}
.main .block-container img[src*="horizontal_logo"],
.main .block-container img[alt*="horizontal_logo"] {
  max-height: 70px !important;
  height: auto !important;
  width: auto !important;
  margin: 0 !important;
  padding: 0 !important;
  margin-bottom: 0 !important;
  display: block !important;
  opacity: 0.4 !important;
  filter: grayscale(20%) !important;
  object-fit: contain !important;
}
/* Remove spacing between logo and subheader */
.main .block-container div[data-testid="stSubheader"] {
  margin-top: 0 !important;
  padding-top: 0 !important;
}


/* File uploader styling (optional) */
.stFileUploader{ background:#FFF1FA; border:1px solid #E0BEEB; border-radius:10px; padding:12px; }

/* Input fields - match Category (selectbox) exactly - dark purple border by default */
/* Remove ALL borders from container divs to prevent double/thick borders - EXTRA AGGRESSIVE */
.stTextInput,
.stTextInput > div,
.stTextInput > div > div,
.stTextInput > div > div > div {
  border: none !important;
  border-width: 0 !important;
  border-color: transparent !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Text input - match selectbox styling exactly */
.stTextInput > div > div > input {
  color: #2E2E2E !important;
  background-color: #FFFFFF !important;
  border: 1px solid #6B4C98 !important;
  border-width: 1px !important;
  border-color: #6B4C98 !important;
  border-radius: 8px !important;
  font-family: inherit !important;
  font-size: 14px !important;
  padding: 0.5rem !important;
  outline: none !important;
  box-shadow: none !important;
}

.stTextArea > div > div > textarea {
  color: #2E2E2E !important;
  background-color: #FFFFFF !important;
  border: 1px solid #6B4C98 !important;
  border-radius: 8px !important;
  font-family: inherit !important;
  font-size: 14px !important;
  padding: 0.5rem !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Textarea container - make it white like other inputs */
.stTextArea > div > div {
  background-color: #FFFFFF !important;
}

.stTextArea > div > div > textarea {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
  border: 1px solid #6B4C98 !important;
  border-radius: 8px !important;
}

/* Fix textarea inner background - more specific */
.stTextArea textarea {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
}

.stTextArea > div textarea {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
}

div[data-baseweb="textarea"] textarea {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
  border: 1px solid #6B4C98 !important;
}

/* Textarea container - remove borders to prevent double borders and fix weird corners */
.stTextArea > div,
.stTextArea > div > div {
  border: none !important;
  border-width: 0 !important;
  outline: none !important;
  box-shadow: none !important;
  overflow: hidden !important;
}

/* Fix textarea corners - remove resizing handle artifacts */
.stTextArea > div > div > textarea {
  resize: vertical !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}

/* Remove corner artifacts from textarea */
div[data-baseweb="textarea"] textarea {
  border-radius: 8px !important;
  resize: vertical !important;
}

/* Change hover/focus border to light purple for all textareas (no red!) */
.stTextArea > div > div > textarea:focus,
.stTextArea textarea:focus,
div[data-baseweb="textarea"] textarea:focus,
.stTextArea > div > div > textarea:focus-visible,
.stTextArea textarea:focus-visible,
.stTextArea > div > div > textarea:hover,
.stTextArea textarea:hover {
  border: 1px solid #E0BEEB !important;
  border-width: 1px !important;
  border-color: #E0BEEB !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Text input focus/hover - light purple, NO RED, NO DOUBLE BORDERS */
.stTextInput > div > div > input:focus,
.stTextInput > div > div > input:hover,
.stTextInput > div > div > input:focus-visible,
.stTextInput > div > div > input:focus-within,
.stTextInput input:focus,
.stTextInput input:hover,
.stTextInput input:focus-visible {
  border: 1px solid #E0BEEB !important;
  border-width: 1px !important;
  border-color: #E0BEEB !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Remove any red/dark borders from text input containers - EXTRA AGGRESSIVE */
.stTextInput,
.stTextInput > div,
.stTextInput > div > div,
.stTextInput > div:focus,
.stTextInput > div:hover,
.stTextInput > div:focus-within,
.stTextInput > div > div:focus,
.stTextInput > div > div:hover,
.stTextInput > div > div:focus-within,
.stTextInput > div > div > div {
  border: none !important;
  border-width: 0 !important;
  border-color: transparent !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Selectbox - dark purple border by default, light purple on hover (NO RED!) */
.stSelectbox > div[data-baseweb="select"] {
  border: 1px solid #6B4C98 !important;
  border-width: 1px !important;
  border-color: #6B4C98 !important;
  outline: none !important;
  box-shadow: none !important;
}

.stSelectbox > div[data-baseweb="select"] > div {
  border: none !important;
  outline: none !important;
}

/* On hover/focus - light purple border, NO RED */
.stSelectbox > div[data-baseweb="select"]:focus,
.stSelectbox > div[data-baseweb="select"]:focus-within,
.stSelectbox > div[data-baseweb="select"]:focus-visible,
.stSelectbox > div[data-baseweb="select"]:hover {
  border: 1px solid #E0BEEB !important;
  border-color: #E0BEEB !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Multiselect - white background, dark purple border by default (NOT BLACK!) - match other fields */
.stMultiSelect > div[data-baseweb="select"] {
  background-color: #FFFFFF !important;
  border: 1px solid #6B4C98 !important;
  border-width: 1px !important;
  border-color: #6B4C98 !important;
  border-radius: 8px !important;
  color: #2E2E2E !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Ensure multiselect border matches other fields exactly */
.stMultiSelect > div[data-baseweb="select"]:not(:focus):not(:hover):not(:focus-within):not(:focus-visible) {
  border: 1px solid #6B4C98 !important;
  border-color: #6B4C98 !important;
}

/* Target ALL inner divs to force white background - using the actual HTML structure */
.stMultiSelect > div[data-baseweb="select"] > div,
.stMultiSelect > div[data-baseweb="select"] > div > div,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"],
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div > div,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div > div > div,
/* Target the specific divs from the HTML structure */
.stMultiSelect div[class*="st-av"],
.stMultiSelect div[class*="st-ay"],
.stMultiSelect div[class*="st-aw"],
.stMultiSelect div[class*="st-ax"],
.stMultiSelect div[class*="st-bt"],
.stMultiSelect div[class*="st-bu"],
.stMultiSelect div[class*="st-bl"],
.stMultiSelect div[class*="st-b4"],
.stMultiSelect div[class*="st-b5"],
.stMultiSelect div[class*="st-bv"],
.stMultiSelect div[class*="st-bw"],
.stMultiSelect div[class*="st-bx"],
.stMultiSelect div[class*="st-by"],
.stMultiSelect div[class*="st-bz"],
.stMultiSelect div[class*="st-c0"],
.stMultiSelect div[class*="st-c1"],
.stMultiSelect div[class*="st-c2"],
.stMultiSelect div[class*="st-c3"],
.stMultiSelect div[class*="st-c4"],
.stMultiSelect div[class*="st-cb"],
.stMultiSelect div[class*="st-cc"],
.stMultiSelect div[class*="st-cd"],
.stMultiSelect div[class*="st-ce"],
.stMultiSelect div[class*="st-du"],
.stMultiSelect div[class*="st-cx"],
.stMultiSelect div[class*="st-cy"],
.stMultiSelect div[class*="st-dv"],
.stMultiSelect div[class*="st-d0"],
.stMultiSelect div[class*="st-dw"] {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
}

/* Fix multiselect dropdown menu background */
.stMultiSelect div[data-baseweb="popover"],
.stMultiSelect div[data-baseweb="popover"] > div,
.stMultiSelect div[data-baseweb="menu"],
.stMultiSelect div[data-baseweb="menu"] > ul {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
  border: 1px solid #E0BEEB !important;
}

/* Fix multiselect input field and placeholder - EXTRA AGGRESSIVE */
.stMultiSelect input,
.stMultiSelect > div[data-baseweb="select"] input,
.stMultiSelect > div[data-baseweb="select"] > div input,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] input,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div input,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div > div input,
/* Target input by class names from HTML */
.stMultiSelect input[class*="st-ae"],
.stMultiSelect input[class*="st-af"],
.stMultiSelect input[class*="st-ag"],
.stMultiSelect input[class*="st-ah"],
.stMultiSelect input[class*="st-ai"],
.stMultiSelect input[class*="st-aj"],
.stMultiSelect input[class*="st-c6"],
.stMultiSelect input[class*="st-d2"],
.stMultiSelect input[class*="st-e0"],
.stMultiSelect input[class*="st-cm"],
.stMultiSelect input[class*="st-d4"],
.stMultiSelect input[class*="st-cs"],
.stMultiSelect input[class*="st-ct"],
.stMultiSelect input[class*="st-cu"],
.stMultiSelect input[class*="st-cv"],
.stMultiSelect input[class*="st-az"],
.stMultiSelect input[class*="st-cq"],
.stMultiSelect input[class*="st-cw"],
.stMultiSelect input[class*="st-as"],
.stMultiSelect input[class*="st-at"],
.stMultiSelect input[class*="st-b7"],
.stMultiSelect input[class*="st-b6"],
.stMultiSelect input[class*="st-cx"],
.stMultiSelect input[class*="st-cy"],
.stMultiSelect input[class*="st-cz"],
.stMultiSelect input[class*="st-d0"] {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
  border: none !important;
}

.stMultiSelect input::placeholder,
.stMultiSelect > div[data-baseweb="select"] input::placeholder {
  color: #999999 !important;
}

/* Fix multiselect placeholder text visibility - EXTRA AGGRESSIVE */
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div[data-baseweb="baseui-select"],
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div[data-baseweb="baseui-select"] > div,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div[data-baseweb="baseui-select"] > div > div,
.stMultiSelect [data-baseweb="baseui-select"],
.stMultiSelect [data-baseweb="baseui-select"] > div,
.stMultiSelect [data-baseweb="baseui-select"] * {
  background-color: #FFFFFF !important;
  color: #999999 !important;
}

/* Make placeholder text visible */
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div[data-baseweb="baseui-select"]::before,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div[data-baseweb="baseui-select"] > div::before {
  color: #999999 !important;
}

/* Fix multiselect selected tags/chips */
.stMultiSelect span[data-baseweb="tag"] {
  background-color: #E8D5E3 !important;
  color: #6B4C98 !important;
  border: 1px solid #B392AC !important;
}

.stMultiSelect span[data-baseweb="tag"] svg {
  color: #6B4C98 !important;
}

/* Multiselect hover/focus - light purple border, NO RED, NO BLACK */
.stMultiSelect > div[data-baseweb="select"]:focus,
.stMultiSelect > div[data-baseweb="select"]:focus-within,
.stMultiSelect > div[data-baseweb="select"]:focus-visible,
.stMultiSelect > div[data-baseweb="select"]:hover {
  border: 1px solid #E0BEEB !important;
  border-width: 1px !important;
  border-color: #E0BEEB !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Force dark purple border on multiselect when not hovered (override any black) */
.stMultiSelect > div[data-baseweb="select"]:not(:focus):not(:hover):not(:focus-within):not(:focus-visible) {
  border: 1px solid #6B4C98 !important;
  border-color: #6B4C98 !important;
}

/* Fix "choose an option" placeholder state - dark purple border, light purple on hover */
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"]:focus,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"]:focus-within,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"]:hover {
  border-color: #E0BEEB !important;
  outline: none !important;
}

/* Remove any red borders from multiselect combobox */
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"],
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] > div {
  border: none !important;
  outline: none !important;
}

/* Force white background on ALL divs inside multiselect - catch everything */
.stMultiSelect > div[data-baseweb="select"] div {
  background-color: #FFFFFF !important;
}

/* Exception: tags/chips should keep their purple background */
.stMultiSelect span[data-baseweb="tag"] {
  background-color: #E8D5E3 !important;
}

.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
  color: #999999 !important;
}

/* Selectbox container - white background, already has border set above */
.stSelectbox > div[data-baseweb="select"] {
  background-color: #FFFFFF !important;
  border-radius: 8px !important;
}

.stSelectbox > div[data-baseweb="select"] > div {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
  font-size: 14px !important;
}

/* Selectbox dropdown menu - make it white to match */
div[data-baseweb="popover"] > div[data-baseweb="menu"],
div[data-baseweb="popover"] > div[data-baseweb="menu"] > ul {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
  border: 1px solid #E0BEEB !important;
  border-radius: 8px !important;
}

div[data-baseweb="popover"] > div[data-baseweb="menu"] > ul > li {
  background-color: #FFFFFF !important;
  color: #2E2E2E !important;
}

div[data-baseweb="popover"] > div[data-baseweb="menu"] > ul > li:hover,
div[data-baseweb="popover"] > div[data-baseweb="menu"] > ul > li[aria-selected="true"] {
  background-color: #F7E9FD !important;
  color: #6B4C98 !important;
}

/* Override hover to light purple (config.toml handles default, but we want light purple on hover) */
div[data-baseweb="select"]:focus,
div[data-baseweb="select"]:focus-within,
div[data-baseweb="select"]:focus-visible,
div[data-baseweb="select"]:hover {
  border-color: #E0BEEB !important;
  border-width: 1px !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Override any black borders on multiselect - force dark purple */
.stMultiSelect > div[data-baseweb="select"] {
  border-color: #6B4C98 !important;
}

/* Ensure all borders are uniform width - no thick borders */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div[data-baseweb="select"],
.stMultiSelect > div[data-baseweb="select"] {
  border-width: 1px !important;
}

/* Remove ALL borders from text input wrapper divs to match Category exactly */
.stTextInput > div,
.stTextInput > div > div,
.stTextInput > div > div > div,
.stTextInput > div > div > div > div {
  border: none !important;
  border-width: 0 !important;
  border-color: transparent !important;
  box-shadow: none !important;
}

/* Ensure text input has clean single border like selectbox - override any inherited borders */
.stTextInput input {
  border: 1px solid #6B4C98 !important;
  border-width: 1px !important;
  border-color: #6B4C98 !important;
  box-shadow: none !important;
}

/* Remove outlines globally */
*:focus {
  outline: none !important;
}

/* Change hover/focus to light purple (config.toml handles default colors) */
input:focus,
input:hover,
input:focus-visible,
textarea:focus,
textarea:hover,
textarea:focus-visible,
select:focus,
select:hover,
select:focus-visible {
  border-color: #E0BEEB !important;
  box-shadow: none !important;
  outline: none !important;
}

/* Text inputs hover to light purple */
.stTextInput input:focus,
.stTextInput input:hover,
.stTextInput input:focus-visible,
.stTextInput input:focus-within {
  border: 1px solid #E0BEEB !important;
  border-width: 1px !important;
  border-color: #E0BEEB !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Override any black borders on multiselect */
.stMultiSelect > div[data-baseweb="select"],
.stMultiSelect > div[data-baseweb="select"]:not(:focus):not(:hover) {
  border-color: #6B4C98 !important;
}

/* Multiselect hover to light purple (where "choose an option" appears) */
.stMultiSelect > div[data-baseweb="select"] input:focus,
.stMultiSelect > div[data-baseweb="select"] input:hover,
.stMultiSelect > div[data-baseweb="select"] input:focus-visible,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"]:focus,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"]:hover,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"]:focus-within {
  border-color: #E0BEEB !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Ensure multiselect input area has no border (parent has the border) */
.stMultiSelect > div[data-baseweb="select"] input,
.stMultiSelect > div[data-baseweb="select"] > div[role="combobox"] input {
  border: none !important;
  outline: none !important;
}

/* Form labels - make labels visible above input boxes (not bold) */
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stMultiSelect label {
  color: #6B4C98 !important;
  font-weight: normal !important;
}

/* Make sure multiselect label is visible - more specific selectors */
.stMultiSelect label,
.stMultiSelect > label,
.stMultiSelect > div > label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stMultiSelect"] > label {
  color: #6B4C98 !important;
  font-weight: normal !important;
  display: block !important;
  visibility: visible !important;
}

/* Tag-style buttons for occasions and seasons - default unselected */
.stButton > button {
  border-radius: 20px !important;
  padding: 6px 14px !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  background-color: #E8D5E3 !important;
  color: #6B4C98 !important;
  border: 1px solid #B392AC !important;
}

.stButton > button:hover {
  background-color: #D1B3C4 !important;
  border-color: #6B4C98 !important;
}

/* Style wardrobe item cards */
.wardrobe-item-card {
  background: white !important;
  border-radius: 12px !important;
  overflow: hidden !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
  margin-bottom: 0 !important;
  display: flex !important;
  flex-direction: column !important;
}

/* Card images */
.wardrobe-item-card img {
  width: 100% !important;
  height: 280px !important;
  object-fit: cover !important;
  object-position: center !important;
  display: block !important;
  margin: 0 !important;
  border-radius: 12px 12px 0 0 !important;
}

/* Main wardrobe grid columns - no card styling on these */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
  background: transparent !important;
  box-shadow: none !important;
  padding: 8px !important;
}

/* Reduce gap between elements inside columns */
div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] > div {
  gap: 0 !important;
}

div[data-testid="column"] > div > div > div[data-testid="element-container"] {
  margin-bottom: 0 !important;
}

/* Card action buttons - Edit and Delete - FORCE visible text */
.wardrobe-item-card + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button,
div[data-testid="column"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
  background: #f7e9fd !important;
  border: 1px solid #B392AC !important;
  border-radius: 6px !important;
  padding: 4px 8px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: #5D3A7A !important;
  box-shadow: none !important;
  width: 100% !important;
  height: 28px !important;
  min-height: 28px !important;
  margin-top: 4px !important;
  white-space: nowrap !important;
  overflow: hidden !important;
}

/* Force button text to be visible - target all inner elements */
div[data-testid="column"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button *,
div[data-testid="column"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button p,
div[data-testid="column"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button span,
div[data-testid="column"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button div {
  color: #5D3A7A !important;
  fill: #5D3A7A !important;
  opacity: 1 !important;
}

div[data-testid="column"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover,
div[data-testid="column"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover * {
  background: #F8F4FA !important;
  border-color: #6B4C98 !important;
  color: #5D3A7A !important;
}

/* Remove extra margins from button containers */
div[data-testid="column"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] {
  margin: 0 !important;
  padding: 0 !important;
}

/* Style the button row container - tight spacing */
div[data-testid="column"] div[data-testid="stHorizontalBlock"] {
  gap: 4px !important;
  margin-top: 0 !important;
  padding: 0 !important;
}

/* Ensure inner button columns don't inherit card styling */
div[data-testid="column"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
  background: transparent !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  padding: 0 2px !important;
}

/* Popover Edit/Delete buttons - Clear by default, dark fill when active */
div[data-testid="column"] div[data-testid="stPopover"] > button,
div[data-testid="column"] div[data-testid="stPopover"] > button:first-child {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  border-radius: 10px !important;
  color: #6B4C98 !important;
  font-size: 14px !important;
  height: 36px !important;
  width: 100% !important;
  min-width: 60px !important;
  text-align: center !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: background-color 0.2s ease !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
/* Active/clicked state */
div[data-testid="column"] div[data-testid="stPopover"] > button:active,
div[data-testid="column"] div[data-testid="stPopover"] > button:focus {
  background: #f7e9fd !important;
  background-color: #f7e9fd !important;
  border: 1px solid #D1B3C4 !important;
}
/* Hover state */
div[data-testid="column"] div[data-testid="stPopover"] > button:hover {
  background: rgba(247, 233, 253, 0.5) !important;
  background-color: rgba(247, 233, 253, 0.5) !important;
}
div[data-testid="column"] div[data-testid="stPopover"] > button p,
div[data-testid="column"] div[data-testid="stPopover"] > button:first-child p {
  color: #6B4C98 !important;
  -webkit-text-fill-color: #6B4C98 !important;
}
div[data-testid="column"] div[data-testid="stPopover"] > button:hover {
  background: #F7E9FD !important;
  border-color: #6B4C98 !important;
}

/* Sidebar buttons - ensure full width */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button:first-child {
  width: 100% !important;
  min-width: 200px !important;
  padding: 0.6em 1em !important;
}

/* Style popover buttons - Match backend connected styling */
div[data-testid="column"] div[data-testid="stPopover"] > button {
  background: #f7e9fd !important;
  background-color: #f7e9fd !important;
  border: 1px solid #D1B3C4 !important;
  border-radius: 10px !important;
  color: #6B4C98 !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  padding: 6px 16px !important;
  height: 36px !important;
  width: 100% !important;
  min-width: 80px !important;
  text-align: center !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
div[data-testid="column"] div[data-testid="stPopover"] > button:hover {
  background: #F7E9FD !important;
  border-color: #6B4C98 !important;
}
div[data-testid="column"] div[data-testid="stPopover"] > button p,
div[data-testid="column"] div[data-testid="stPopover"] > button span {
  color: #6B4C98 !important;
  -webkit-text-fill-color: #6B4C98 !important;
}

/* Hide ONLY the expand_more Material icon - target by data-testid */
div[data-testid="stPopover"] span[data-testid="stIconMaterial"] {
  display: none !important;
}

div[data-testid="column"] div[data-testid="stPopover"] > button:hover {
  background: #F7E9FD !important;
  border-color: #6B4C98 !important;
}

/* Sidebar buttons styling already handled above */

/* Style the popover content panel */
div[data-testid="stPopoverBody"] {
  background: #FFFFFF !important;
  border: 1px solid #E8D5E3 !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 24px rgba(107, 76, 152, 0.2) !important;
  padding: 16px !important;
}

/* ========== FINAL OVERRIDES ========== */

/* Popover trigger buttons (Edit/Delete) - Match backend connected styling */
div[data-testid="stPopover"] button,
div[data-testid="stPopover"] > button,
div[data-testid="stPopover"] > div > button,
[data-testid="stPopover"] button {
  background: #f7e9fd !important;
  background-color: #f7e9fd !important;
  border: 1px solid #D1B3C4 !important;
  border-radius: 10px !important;
  color: #6B4C98 !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  height: 36px !important;
  width: 100% !important;
  min-width: 80px !important;
  text-align: center !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
div[data-testid="stPopover"] button:hover,
[data-testid="stPopover"] button:hover {
  background: #F7E9FD !important;
  background-color: #F7E9FD !important;
  border-color: #6B4C98 !important;
}
div[data-testid="stPopover"] button p,
div[data-testid="stPopover"] button span,
[data-testid="stPopover"] button p,
[data-testid="stPopover"] button span {
  color: #6B4C98 !important;
  -webkit-text-fill-color: #6B4C98 !important;
}
/* But still hide the expand icon */
div[data-testid="stPopover"] span[data-testid="stIconMaterial"],
[data-testid="stPopover"] span[data-testid="stIconMaterial"] {
  display: none !important;
}

/* Saved Outfits Edit/Delete buttons - match wardrobe card styling */
/* Only target buttons in main content area, not sidebar */
section[data-testid="stMain"] div[data-testid="stButton"] button[kind="secondary"] {
  background: #f7e9fd !important;
  background-color: #f7e9fd !important;
  border: 1px solid #D1B3C4 !important;
  border-radius: 10px !important;
  color: #6B4C98 !important;
  font-size: 15px !important;
  font-weight: 500 !important;
}
section[data-testid="stMain"] div[data-testid="stButton"] button[kind="secondary"]:hover {
  background: #F7E9FD !important;
  border-color: #6B4C98 !important;
}
section[data-testid="stMain"] div[data-testid="stButton"] button[kind="secondary"] p,
section[data-testid="stMain"] div[data-testid="stButton"] button[kind="secondary"] span {
  color: #6B4C98 !important;
}

/* Dialog X remove buttons only - not Save/Cancel (which use type="primary") */
div[data-testid="stDialog"] div[data-testid="stButton"] button[kind="secondary"] {
  background: #f7e9fd !important;
  border: 1px solid #D1B3C4 !important;
  border-radius: 10px !important;
  color: #6B4C98 !important;
  font-size: 16px !important;
  font-weight: 500 !important;
}
div[data-testid="stDialog"] div[data-testid="stButton"] button[kind="secondary"]:hover {
  background: #F7E9FD !important;
  border-color: #6B4C98 !important;
}
div[data-testid="stDialog"] div[data-testid="stButton"] button[kind="secondary"] p {
  color: #6B4C98 !important;
}

/* Ensure Material Icons font is used for icon elements */
span[class*="material"],
span[class*="Material"],
span[data-testid*="Icon"],
div[data-testid*="Icon"] {
  font-family: 'Material Icons', 'Nunito', sans-serif !important;
}

/* Hide icon text that appears as fallback when Material Icons font fails */
span[data-testid*="IconMaterial"]:not([data-testid="stIconMaterial"]) {
  display: none !important;
  visibility: hidden !important;
  font-size: 0 !important;
}

/* Hide any text content that matches Material icon names (like keyboard_double_arrow_right) */
.stApp * {
  text-rendering: optimizeLegibility;
}

/* Specifically target and hide Material icon name text */
span[class*="material-icons"],
.material-icons {
  font-family: 'Material Icons' !important;
  font-weight: normal !important;
  font-style: normal !important;
  font-size: 24px !important;
  line-height: 1 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  display: inline-block !important;
  white-space: nowrap !important;
  word-wrap: normal !important;
  direction: ltr !important;
  -webkit-font-feature-settings: 'liga' !important;
  -webkit-font-smoothing: antialiased !important;
}
</style>
""",
    unsafe_allow_html=True,
)



# --- Buttons on Outfit Builder and Saved Outfits page ---
st.markdown(
    """
<style>
/* Recolor Streamlit alerts (incl. st.info) to lilac with readable text */
.stAlert, div[role="alert"], div[data-baseweb="notification"] {
  background:#F7E9FD !important;
  border:1px solid #D1B3C4 !important;
  border-radius:10px !important;
  color:#6B4C98 !important;
}
.stAlert *, div[role="alert"] *, div[data-baseweb="notification"] * {
  color:#6B4C98 !important;
  fill:#6B4C98 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# --- Initialize API Client ---
@st.cache_resource
def get_api_client():
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    return APIClient(base_url=backend_url)

api_client = get_api_client()

# --- Cached API Functions ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_saved_outfits():
    # Get saved outfits with caching
    return api_client.get_saved_outfits()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_outfits(occasion: str, season: str):
    # Generate outfits with caching based on occasion and season
    return api_client.generate_outfits(occasion=occasion, season=season)

@st.cache_data(ttl=60)  # Cache for 1 minute (shorter TTL for presigned URLs)
def get_cached_wardrobe_items():
    # Get wardrobe items from database with caching
    # Shorter TTL because presigned URLs expire after 1 hour
    try:
        return api_client.get_wardrobe_items()
    except Exception as e:
        # Re-raise with more context
        raise Exception(f"Failed to fetch wardrobe items: {str(e)}")

# --- Sidebar ---
st.sidebar.image("logo.png")
st.sidebar.title("Generate outfits from your wardrobe.")
#st.sidebar.write("Your digital stylist")

if "page" not in st.session_state:
    st.session_state.page = "My Wardrobe"

# --- Sidebar buttons with active state ---
pages = ["My Wardrobe", "Outfit Builder", "Saved Outfits"]

for page_name in pages:
    is_active = st.session_state.page == page_name
    
    if is_active:
        # Active button - darker purple
        st.sidebar.markdown(
            f"""
            <div style="
                background: #715d78;
                color: white;
                padding: 0.6em 1em;
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                margin: 6px 0;
                cursor: default;
            ">{page_name}</div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Inactive button - clickable
        if st.sidebar.button(page_name, use_container_width=True, key=f"nav_{page_name}"):
            st.session_state.page = page_name
            st.rerun()

page = st.session_state.page

# --- Initialize session state (MUST be first) ---
if "uploaded_items" not in st.session_state:
    st.session_state.uploaded_items = []
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "saved_outfits" not in st.session_state:
    st.session_state.saved_outfits = []
if "outfit_feedback" not in st.session_state:
    st.session_state.outfit_feedback = {}
if "items_loaded_from_backend" not in st.session_state:
    st.session_state.items_loaded_from_backend = False
if "editing_item_id" not in st.session_state:
    st.session_state.editing_item_id = None
if "show_save_success" not in st.session_state:
    st.session_state.show_save_success = False
if "editing_outfit" not in st.session_state:
    st.session_state.editing_outfit = None

# --- My Wardrobe Page ---
if page == "My Wardrobe":
    # Load items from backend on first load (only once per session)
    if not st.session_state.items_loaded_from_backend:
        try:
            response = get_cached_wardrobe_items()
            backend_items = response.get("items", [])

            
            # Convert backend items to match session state format
            # Backend items have image_url, but need to keep them as-is
            # Newly uploaded items will have image_base64 for immediate display
            for backend_item in backend_items:
                item_id = backend_item.get("item_id")
                # Check if item already exists in session state (by item_id)
                existing_item = None
                existing_idx = None
                for idx, existing in enumerate(st.session_state.uploaded_items):
                    if existing.get("item_id") == item_id:
                        existing_item = existing
                        existing_idx = idx
                        break
                
                if existing_item:
                    # Update with backend data but preserve image_base64 for display
                    image_base64 = existing_item.get("image_base64")
                    # Update all fields from backend (includes edited metadata)
                    for key, value in backend_item.items():
                        existing_item[key] = value
                    # Restore image_base64 if it existed (for immediate display)
                    if image_base64:
                        existing_item["image_base64"] = image_base64
                else:
                    # New item from backend - add it
                    st.session_state.uploaded_items.append(backend_item)
            
            st.session_state.items_loaded_from_backend = True
        except Exception as e:
            # Show error in sidebar for debugging
            st.sidebar.warning(f"⚠️ Could not load items from backend: {str(e)}")
            # Continue with session state items (might be empty)
    
    # Logo directly above My Wardrobe with no spacing
    st.image("horizontal_logo.png", use_container_width=False, width=400)
    st.subheader("My Wardrobe")
    
    # Show success message if item was just saved (auto-dismisses)
    if st.session_state.show_save_success:
        st.toast("Changes saved!")
        st.session_state.show_save_success = False
    
    item_count = len(st.session_state.uploaded_items) if st.session_state.uploaded_items else 0
    if item_count == 1:
        st.write(f"{item_count} item in your collection")
    else:
        st.write(f"{item_count} items in your collection")

    col_add = st.columns([5, 1])[1]
    with col_add:
        if st.button("+ Add New Item", type="primary"):
            st.session_state.show_uploader = True

    if len(st.session_state.uploaded_items) == 0 and not st.session_state.show_uploader:
        st.markdown(
            """
            <div style='text-align:center; margin-top:60px;'>
                <svg width="100" height="100" fill="none" stroke="gray" stroke-width="2">
                    <rect x="30" y="30" width="40" height="40"/>
                    <path d="M50 30 L50 70" stroke="gray"/>
                </svg>
                <h2>Your Wardrobe is empty</h2>
                <p>Start building your digital closet by uploading photos of your clothing items.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # Show "Upload Your First Item" button when uploader is not shown and wardrobe is empty
    if len(st.session_state.uploaded_items) == 0 and not st.session_state.show_uploader:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Upload Your First Item!", type="primary", use_container_width=True):
                st.session_state.show_uploader = True
                st.rerun()

    if st.session_state.show_uploader:
        st.markdown("---")
        # Show "Add First Item" if wardrobe is empty, otherwise just show uploader
        if len(st.session_state.uploaded_items) == 0:
            st.markdown("### Add First Item")
        else:
            st.markdown("### Add New Item")
        st.caption("Upload a photo and we'll help you catalog it")

        uploaded_file = st.file_uploader(
            label="",  # Empty label to avoid double text
            type=["png", "jpg", "jpeg", "webp"]
        )

        if uploaded_file:
            st.image(uploaded_file, width=300)
            st.markdown("### Item Details")

            category = st.selectbox(
                "Category *", ["Tops", "Bottoms", "Dresses", "Outerwear", "Shoes", "Accessories"]
            )
            subcategory = st.text_input("Subcategory", placeholder="e.g., tank top")
            season = st.multiselect(
                "Season *",
                ["Spring", "Summer", "Fall", "Winter", "All-Season"],
                default=["All-Season"]
            )
            
            brand = st.text_input("Brand", placeholder="e.g., Christopher Esber")
            colors = st.text_input(
                "Colors (comma-separated)", placeholder="e.g., brown, beige"
            )
            
            occasions = st.multiselect(
                "Occasion *",
                ["Any Occasion", "Casual", "Formal", "Business", "Athletic", "Party", "Everyday"],
                default=["Casual"]
            )
            notes = st.text_area(
                "Notes", placeholder="Add any personal notes about this item"
            )
            
            # Add required field indicator at bottom
            st.markdown("<p style='color: #6B4C98; font-size: 0.875rem;'>* Required fields</p>", unsafe_allow_html=True)

            col1, col2 = st.columns(2, gap="small")
            with col1:
                if st.button("Add to Wardrobe", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Uploading item and computing embeddings..."):
                            # Read file bytes once and reuse
                            uploaded_file.seek(0)
                            file_bytes = uploaded_file.read()
                            
                            # Validate we have bytes
                            if not file_bytes or len(file_bytes) == 0:
                                st.error("Failed to read image file")
                                st.stop()
                            
                            file_bytes_io = BytesIO(file_bytes)
                            
                            # Upload to backend (creates item with just category)
                            response = api_client.upload_wardrobe_item(
                                image_file=file_bytes_io,
                                filename=uploaded_file.name,
                                category=category
                            )
                            
                            # Ensure required fields have defaults
                            final_season = season if season else ["All-Season"]
                            final_occasions = occasions if occasions else ["Casual"]
                            colors_list = [c.strip() for c in colors.split(",") if c.strip()] if colors else []
                            
                            # Update item with all metadata (season, occasion, etc.)
                            item_id = response.get("item_id")
                            if item_id:
                                api_client.update_wardrobe_item(
                                    item_id=item_id,
                                    category=category,
                                    subcategory=subcategory or "",
                                    season=final_season,
                                    brand=brand or "",
                                    colors=colors_list,
                                    occasions=final_occasions,
                                    notes=notes or ""
                                )
                            
                            # Store in session state for display (until backend has GET endpoint)
                            # Use base64 encoding for reliable storage in session state
                            # Verify bytes are valid image data first
                            try:
                                test_image = Image.open(BytesIO(file_bytes))
                                test_image.verify()  # Verify it's a valid image
                            except Exception as e:
                                st.warning(f"Image validation warning: {str(e)}")
                            
                            # Convert bytes to base64 for reliable session state storage
                            image_base64 = base64.b64encode(file_bytes).decode('utf-8')
                            
                            # Verify base64 can be decoded back (sanity check)
                            try:
                                test_decode = base64.b64decode(image_base64)
                                if len(test_decode) != len(file_bytes):
                                    st.warning("Base64 encoding verification failed - size mismatch")
                            except Exception as e:
                                st.error(f"Base64 encoding error: {str(e)}")
                            
                            item_data = {
                                "item_id": response.get("item_id"),
                                "image_url": response.get("image_url"),
                                "category": category,
                                "season": final_season,
                                "subcategory": subcategory,
                                "brand": brand,
                                "colors": colors_list,
                                "occasions": final_occasions,
                                "notes": notes,
                                "image_base64": image_base64  # Store as base64 string
                            }
                            
                            if "uploaded_items" not in st.session_state:
                                st.session_state.uploaded_items = []
                            # Insert at beginning so newest items appear first
                            st.session_state.uploaded_items.insert(0, item_data)
                            
                            # Clear cache so backend items refresh
                            get_cached_wardrobe_items.clear()
                            
                            st.session_state.show_uploader = False
                            st.success(f"Added '{brand or 'New Item'}' to your wardrobe! (ID: {response.get('item_id')})")
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Failed to upload item: {str(e)}")
            with col2:
                if st.button("Cancel", type="primary", use_container_width=True):
                    st.session_state.show_uploader = False
                    st.rerun()

    elif len(st.session_state.uploaded_items) > 0:
        # Initialize filter state
        if "filter_category" not in st.session_state:
            st.session_state.filter_category = "All Categories"
        if "filter_occasion" not in st.session_state:
            st.session_state.filter_occasion = "All Occasions"
        if "filter_color" not in st.session_state:
            st.session_state.filter_color = "All Colors"
        if "filter_season" not in st.session_state:
            st.session_state.filter_season = "All Seasons"
        
        # Get unique values for filters
        all_categories = ["All Categories"] + sorted(list(set([item.get("category", "") for item in st.session_state.uploaded_items if item.get("category")])))
        all_occasions = ["All Occasions"] + sorted(list(set([occ for item in st.session_state.uploaded_items for occ in item.get("occasions", [])])))
        all_colors = ["All Colors"] + sorted(list(set([color for item in st.session_state.uploaded_items for color in item.get("colors", [])])))
        all_seasons = ["All Seasons"] + sorted(list(set([season for item in st.session_state.uploaded_items for season in (item.get("season", []) if isinstance(item.get("season"), list) else [item.get("season")] if item.get("season") else [])])))
        
        # Filter dropdowns - 2 rows for better alignment at all zoom levels
        filter_row1_col1, filter_row1_col2, filter_row1_col3, filter_row1_col4 = st.columns(4)
        with filter_row1_col1:
            st.session_state.filter_category = st.selectbox(
                "Filter by Category",
                all_categories,
                index=all_categories.index(st.session_state.filter_category) if st.session_state.filter_category in all_categories else 0
            )
        with filter_row1_col2:
            st.session_state.filter_occasion = st.selectbox(
                "Filter by Occasion",
                all_occasions,
                index=all_occasions.index(st.session_state.filter_occasion) if st.session_state.filter_occasion in all_occasions else 0
            )
        with filter_row1_col3:
            st.session_state.filter_color = st.selectbox(
                "Filter by Color",
                all_colors,
                index=all_colors.index(st.session_state.filter_color) if st.session_state.filter_color in all_colors else 0
            )
        with filter_row1_col4:
            st.session_state.filter_season = st.selectbox(
                "Filter by Season",
                all_seasons,
                index=all_seasons.index(st.session_state.filter_season) if st.session_state.filter_season in all_seasons else 0
            )
        
        # Filter items - only apply filter if not "All" option
        filtered_items = st.session_state.uploaded_items.copy()
        
        # Category filter
        filter_cat = str(st.session_state.filter_category).strip() if st.session_state.filter_category else ""
        if filter_cat and filter_cat != "All Categories":
            filtered_items = [item for item in filtered_items if item.get("category") == filter_cat]
        
        # Occasion filter - include "Any Occasion" items when filtering by specific occasion
        filter_occ = str(st.session_state.filter_occasion).strip() if st.session_state.filter_occasion else ""
        if filter_occ and filter_occ != "All Occasions":
            item_occasions = lambda item: item.get("occasions") or []
            filtered_items = [item for item in filtered_items if filter_occ in item_occasions(item) or "Any Occasion" in item_occasions(item)]
        
        # Color filter - explicitly skip if "All Colors" is selected
        filter_color = str(st.session_state.filter_color).strip() if st.session_state.filter_color else ""
        if filter_color and filter_color != "All Colors":
            # Only filter by color if NOT "All Colors"
            filtered_items = [item for item in filtered_items if filter_color.lower() in [c.lower() for c in item.get("colors", [])]]
        # If "All Colors" is selected, don't filter by color (show all items regardless of color)
        
        # Season filter
        filter_season = str(st.session_state.filter_season).strip() if st.session_state.filter_season else ""
        if filter_season and filter_season != "All Seasons":
            def item_has_season(item, season):
                item_seasons = item.get("season")
                # Handle None, missing key, or empty values
                if item_seasons is None:
                    return False
                if isinstance(item_seasons, str):
                    item_seasons = [item_seasons] if item_seasons else []
                if not isinstance(item_seasons, list):
                    return False
                # Include "All-Season" items when filtering by any specific season
                return season in item_seasons or "All-Season" in item_seasons
            filtered_items = [item for item in filtered_items if item_has_season(item, filter_season)]
        
        # Display filtered items in card layout using st.columns for grid
        if filtered_items:
            # Create grid: 3 columns per row for better spacing
            items_per_row = 3
            for row_start in range(0, len(filtered_items), items_per_row):
                row_items = filtered_items[row_start:row_start + items_per_row]
                cols = st.columns(len(row_items))
                
                for col_idx, item in enumerate(row_items):
                    with cols[col_idx]:
                        item_id = item.get('item_id', row_start + col_idx)
                        
                        # Get image as base64 for embedding
                        image_src = ""
                        image_base64 = item.get("image_base64")
                        if image_base64:
                            image_src = f"data:image/jpeg;base64,{image_base64}"
                        else:
                            image_url = item.get("image_url")
                            if image_url and isinstance(image_url, str) and image_url.strip() and not image_url.startswith("s3://"):
                                image_src = image_url
                        
                        # Item details
                        subcategory = item.get('subcategory')
                        category = item.get('category')
                        brand = item.get('brand')
                        item_name = subcategory or category or brand or 'Item'
                        if item_name != 'Item':
                            item_name = str(item_name).title()
                        
                        # Image tag
                        image_tag = f'<img src="{image_src}" style="width:100%; height:280px; object-fit:cover; object-position:center; display:block; margin:0; border-radius:12px 12px 0 0;" />' if image_src else '<div style="background:#f5f5f5; height:280px; display:flex; align-items:center; justify-content:center; color:#999; border-radius:12px 12px 0 0;">No image</div>'
                        
                        # Open card container
                        st.markdown(f'<div class="wardrobe-item-card" id="card_{item_id}">', unsafe_allow_html=True)
                        
                        # Image
                        st.markdown(image_tag, unsafe_allow_html=True)
                        
                        # Content container - always reserve space for notes to keep cards aligned
                        notes = item.get("notes", "")
                        st.markdown(
                            f'<div style="padding:10px 12px 4px 12px;">'
                            f'<h4 style="margin:0 0 2px 0; color:#2E2E2E; font-size:15px; font-weight:600;">{item_name}</h4>'
                            f'<div style="min-height:18px; margin-bottom:2px; color:#666; font-size:11px; font-style:italic; line-height:1.2;">{notes if notes else ""}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # All tags in one row with uniform styling
                        all_tags = ""
                        tag_style = "background:#E8D5E3; color:#6B4C98; padding:3px 8px; border-radius:10px; font-size:10px; margin-right:4px; margin-bottom:4px; display:inline-block;"
                        grey_tag_style = "background:#f5f5f5; color:#666; padding:3px 8px; border-radius:10px; font-size:10px; margin-right:4px; margin-bottom:4px; display:inline-block;"
                        
                        # Category tag (purple)
                        if item.get("category"):
                            all_tags += f'<span style="{tag_style}">{item.get("category")}</span>'
                        
                        # Subcategory tag (grey)
                        if item.get("subcategory"):
                            all_tags += f'<span style="{grey_tag_style}">{item.get("subcategory")}</span>'
                        
                        # Season tags (handle both old single value and new list format)
                        season_value = item.get("season", [])
                        if season_value:
                            if isinstance(season_value, str):
                                # Old format: single string
                                all_tags += f'<span style="{grey_tag_style}">{season_value}</span>'
                            elif isinstance(season_value, list):
                                # New format: list
                                for season in season_value:
                                    all_tags += f'<span style="{grey_tag_style}">{season}</span>'
                        
                        # Brand tag (grey)
                        if item.get("brand"):
                            all_tags += f'<span style="{grey_tag_style}">{item.get("brand")}</span>'
                        
                        # Color tags (grey)
                        for color in item.get("colors", []):
                            all_tags += f'<span style="{grey_tag_style}">{color}</span>'
                        
                        # Occasion tags (grey)
                        for occ in item.get("occasions", []):
                            all_tags += f'<span style="{grey_tag_style}">{occ}</span>'
                        
                        st.markdown(
                            f'<div style="display:flex; flex-wrap:wrap; align-items:flex-start; padding:0 12px 8px 12px; min-height:70px; max-height:70px; overflow:hidden;">{all_tags}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Close card HTML
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Buttons row with popovers (consistent spacing)
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            with st.popover("Edit", use_container_width=True):
                                st.markdown("#### Edit Item")
                                edit_category = st.selectbox(
                                    "Category *", 
                                    ["Tops", "Bottoms", "Dresses", "Outerwear", "Shoes", "Accessories"],
                                    index=["Tops", "Bottoms", "Dresses", "Outerwear", "Shoes", "Accessories"].index(item.get("category", "Tops")) if item.get("category") in ["Tops", "Bottoms", "Dresses", "Outerwear", "Shoes", "Accessories"] else 0,
                                    key=f"edit_cat_{item_id}"
                                )
                                edit_subcategory = st.text_input("Subcategory", value=item.get("subcategory", ""), key=f"edit_sub_{item_id}")
                                # Handle season - convert old single value to list if needed
                                season_value = item.get("season", [])
                                if isinstance(season_value, str):
                                    season_value = [season_value] if season_value else []
                                edit_season = st.multiselect(
                                    "Season *",
                                    ["Spring", "Summer", "Fall", "Winter", "All-Season"],
                                    default=season_value if season_value else ["All-Season"],
                                    key=f"edit_season_{item_id}"
                                )
                                edit_brand = st.text_input("Brand", value=item.get("brand", ""), key=f"edit_brand_{item_id}")
                                colors_str = ", ".join(item.get("colors", [])) if item.get("colors") else ""
                                edit_colors = st.text_input("Colors (comma-separated)", value=colors_str, key=f"edit_colors_{item_id}")
                                edit_occasions = st.multiselect(
                                    "Occasion *",
                                    ["Any Occasion", "Casual", "Formal", "Business", "Athletic", "Party", "Everyday"],
                                    default=item.get("occasions", []) if item.get("occasions") else ["Casual"],
                                    key=f"edit_occ_{item_id}"
                                )
                                edit_notes = st.text_area("Notes", value=item.get("notes", ""), key=f"edit_notes_{item_id}", height=60)
                                
                                if st.button("Save", key=f"save_{item_id}", use_container_width=True, type="primary"):
                                    try:
                                        with st.spinner("Saving changes..."):
                                            # Parse colors from comma-separated string, ensure no None values
                                            colors_list = [c.strip() for c in (edit_colors or "").split(",") if c.strip()]

                                            # Ensure required fields have defaults
                                            final_season = edit_season if edit_season else ["All-Season"]
                                            final_occasions = edit_occasions if edit_occasions else ["Casual"]
                                            
                                            # Save to database
                                            api_client.update_wardrobe_item(
                                                item_id=item_id,
                                                category=edit_category,
                                                subcategory=edit_subcategory or "",
                                                season=final_season,
                                                brand=edit_brand or "",
                                                colors=colors_list,
                                                occasions=final_occasions,
                                                notes=edit_notes or ""
                                            )

                                            # Clear cache and force reload from backend on next run
                                            get_cached_wardrobe_items.clear()
                                            st.session_state.items_loaded_from_backend = False
                                            st.session_state.show_save_success = True
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to save: {str(e)}")
                        
                        with btn_col2:
                            with st.popover("Delete", use_container_width=True):
                                st.markdown("#### Delete Item?")
                                st.write("This will permanently remove the item.")
                                if st.button("Yes, Delete", key=f"confirm_delete_{item_id}", use_container_width=True):
                                    try:
                                        # Delete from database
                                        api_client.delete_wardrobe_item(item_id)
                                        # Remove from session state
                                        st.session_state.uploaded_items = [it for it in st.session_state.uploaded_items if it.get('item_id') != item_id]
                                        get_cached_wardrobe_items.clear()
                                        st.success("Item deleted!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Delete failed: {str(e)}")
        else:
            st.info("No items match the selected filters.")

# --- Outfit Builder Page ---
elif page == "Outfit Builder":
    st.subheader("Outfit Builder")
    st.write("Create an outfit by selecting an occasion and season and we will auto-generate one for you!")

    if len(st.session_state.uploaded_items) == 0:
        st.info("Upload some items in 'My Wardrobe' first!")
    else:
        # Initialize outfit selections in session state
        if "outfit_selections" not in st.session_state:
            st.session_state.outfit_selections = {}
        if "generated_outfits" not in st.session_state:
            st.session_state.generated_outfits = None
        
        # --- Auto-Generate Section ---
        gen_col1, gen_col2, gen_col3 = st.columns([2, 2, 1])
        with gen_col1:
            gen_occasion = st.selectbox("Occasion", ["Casual", "Formal", "Business", "Party", "Everyday", "Athletic"], key="gen_occasion")
        with gen_col2:
            gen_season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], key="gen_season")
        with gen_col3:
            st.write("")  # Spacer
            if st.button("Generate", use_container_width=True, type="primary"):
                with st.spinner("Creating outfit suggestions..."):
                    try:
                        response = api_client.generate_outfits(gen_occasion, gen_season)
                        st.session_state.generated_outfits = response.get("outfits", [])
                        if not st.session_state.generated_outfits:
                            st.warning("No outfits found for this combination. Try different options!")
                    except Exception as e:
                        st.error(f"Failed to generate outfits: {e}")
        
        # Display generated outfits
        if st.session_state.generated_outfits:
            st.markdown("#### Suggested Outfits")
            st.caption("Click 'Use This' to select an outfit")
            
            # Create a lookup for items by ID
            items_by_id = {item.get("item_id"): item for item in st.session_state.uploaded_items}
            
            # Display up to 3 outfit suggestions
            outfit_cols = st.columns(min(len(st.session_state.generated_outfits), 3))
            for idx, outfit in enumerate(st.session_state.generated_outfits[:3]):
                with outfit_cols[idx]:
                    score = outfit.get("score", 0)
                    score_display = f"Match: {score:.0%}" if score > 0 else ""
                    st.markdown(f"**Outfit {idx + 1}**")
                    if score_display:
                        st.caption(score_display)
                    outfit_items = []
                    for item_id in outfit.get("items", []):
                        item = items_by_id.get(item_id)
                        if item:
                            outfit_items.append(item)
                            img_src = ""
                            if item.get("image_base64"):
                                img_src = f"data:image/jpeg;base64,{item.get('image_base64')}"
                            elif item.get("image_url") and not item.get("image_url", "").startswith("s3://"):
                                img_src = item.get("image_url")
                            
                            if img_src:
                                st.markdown(
                                    f'<div style="border-radius:8px; overflow:hidden; margin-bottom:4px;">'
                                    f'<img src="{img_src}" style="width:100%; height:80px; object-fit:cover;" />'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            st.caption(f"{item.get('category', '')} - {item.get('subcategory', '') or item.get('brand', '')}")
                    
                    if outfit_items and st.button("Use This", key=f"use_outfit_{idx}", use_container_width=True, type="primary"):
                        # Apply this outfit to selections
                        st.session_state.outfit_selections = {}
                        for item in outfit_items:
                            cat = item.get("category")
                            if cat:
                                st.session_state.outfit_selections[cat] = item.get("item_id")
                        st.session_state.generated_outfits = None  # Clear suggestions
                        st.rerun()
        
        st.markdown("---")
        
        # Helper function to get image source
        def get_image_src(item):
            if item.get("image_base64"):
                return f"data:image/jpeg;base64,{item.get('image_base64')}"
            elif item.get("image_url") and not item.get("image_url", "").startswith("s3://"):
                return item.get("image_url")
            return ""
        
        # Show outfit preview (from selected generated outfit)
        selected_items = []
        for cat, item_id in st.session_state.outfit_selections.items():
            item = next((i for i in st.session_state.uploaded_items if i.get("item_id") == item_id), None)
            if item:
                selected_items.append(item)
        
        if selected_items:
            st.markdown("### Your Outfit")
            st.caption("Click ✕ to remove an item from the outfit")
            
            preview_cols = st.columns(len(selected_items))
            for idx, item in enumerate(selected_items):
                with preview_cols[idx]:
                    img_src = get_image_src(item)
                    if img_src:
                        st.image(img_src, use_container_width=True)
                    st.caption(f"{item.get('category')} - {item.get('subcategory', '')}")
                    
                    # Delete button to remove this item from outfit
                    cat = item.get('category')
                    if st.button("✕", key=f"remove_outfit_item_{cat}_{idx}", use_container_width=True):
                        if cat in st.session_state.outfit_selections:
                            del st.session_state.outfit_selections[cat]
                        st.rerun()
            
            # Add from Wardrobe button
            with st.expander("Add from Wardrobe", expanded=False):
                st.caption("Select an item to add to your outfit")
                
                # Get item IDs already in outfit
                item_ids_in_outfit = set(st.session_state.outfit_selections.values())
                
                # Filter to show items not already in outfit (by ID, so user can swap same-category items)
                available_items = [
                    item for item in st.session_state.uploaded_items 
                    if item.get("item_id") not in item_ids_in_outfit
                ]
                
                if available_items:
                    # Group by category
                    categories = sorted(set(item.get("category", "Other") for item in available_items))
                    selected_category = st.selectbox("Filter by Category", ["All"] + categories, key="add_item_category_filter")
                    
                    if selected_category != "All":
                        available_items = [item for item in available_items if item.get("category") == selected_category]
                    
                    # Display items in grid
                    add_cols = st.columns(4)
                    for idx, item in enumerate(available_items):
                        with add_cols[idx % 4]:
                            img_src = get_image_src(item)
                            if img_src:
                                st.image(img_src, use_container_width=True)
                            st.caption(f"{item.get('category', '')} - {item.get('subcategory', '') or item.get('brand', '')}")
                            if st.button("Add", key=f"add_to_outfit_{item.get('item_id')}", use_container_width=True, type="primary"):
                                cat = item.get("category")
                                st.session_state.outfit_selections[cat] = item.get("item_id")
                                st.rerun()
                else:
                    st.info("All categories already have an item in the outfit, or your wardrobe is empty.")
            
            # Outfit name and details for saving
            st.markdown("#### Save This Outfit")
            outfit_name = st.text_input("Outfit Name", placeholder="e.g., Casual Friday, Date Night, Work Meeting", key="outfit_name")
            save_col1, save_col2 = st.columns(2)
            
            # Get the occasion/season used to generate (default to first option if not set)
            occasion_options = ["Casual", "Formal", "Business", "Party", "Everyday"]
            season_options = ["Spring", "Summer", "Fall", "Winter", "All-Season"]
            
            # Use the generation values as defaults
            default_occasion = st.session_state.get("gen_occasion", "Casual")
            default_season = st.session_state.get("gen_season", "Spring")
            
            occasion_idx = occasion_options.index(default_occasion) if default_occasion in occasion_options else 0
            season_idx = season_options.index(default_season) if default_season in season_options else 0
            
            with save_col1:
                save_occasion = st.selectbox("Occasion", occasion_options, index=occasion_idx, key="save_occasion")
            with save_col2:
                save_season = st.selectbox("Season", season_options, index=season_idx, key="save_season")
            
            if st.button("Save Outfit", use_container_width=True, type="primary"):
                try:
                    item_ids = [item.get("item_id") for item in selected_items]
                    response = api_client.save_outfit(
                        item_ids=item_ids,
                        occasion=save_occasion,
                        season=save_season,
                        name=outfit_name or "My Outfit"
                    )
                    get_cached_saved_outfits.clear()
                    st.success("Outfit saved!")
                    st.session_state.outfit_selections = {}
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save outfit: {str(e)}")
            
            if st.button("Clear Selections", use_container_width=True, type="primary"):
                st.session_state.outfit_selections = {}
                st.rerun()
        else:
            st.info("Generate an outfit above and click 'Use This' to select it!")

# --- Saved Outfits Page ---
elif page == "Saved Outfits":
    st.subheader("Saved Outfits")
    
    # Edit Outfit Dialog
    @st.dialog("Edit Outfit", width="large")
    def edit_outfit_dialog(outfit_data):
        outfit_id = outfit_data.get("outfit_id")
        current_items = outfit_data.get("items", [])
        
        # Get item IDs that are currently in the outfit
        current_item_ids = [item.get("item_id") for item in current_items]
        
        # Track items to keep (start with all current items)
        if f"edit_items_{outfit_id}" not in st.session_state:
            st.session_state[f"edit_items_{outfit_id}"] = current_item_ids.copy()
        
        items_to_keep = st.session_state[f"edit_items_{outfit_id}"]
        
        # Outfit name
        new_name = st.text_input(
            "Outfit Name",
            value=outfit_data.get("name", ""),
            key=f"edit_name_{outfit_id}"
        )
        
        # Occasion and Season
        col1, col2 = st.columns(2)
        with col1:
            occasions = ["Casual", "Formal", "Business", "Party", "Everyday"]
            current_occasion = outfit_data.get("occasion", "Casual")
            occasion_idx = occasions.index(current_occasion) if current_occasion in occasions else 0
            new_occasion = st.selectbox(
                "Occasion",
                occasions,
                index=occasion_idx,
                key=f"edit_occasion_{outfit_id}"
            )
        with col2:
            seasons = ["Spring", "Summer", "Fall", "Winter", "All-Season"]
            current_season = outfit_data.get("season", "All-Season")
            season_idx = seasons.index(current_season) if current_season in seasons else 0
            new_season = st.selectbox(
                "Season",
                seasons,
                index=season_idx,
                key=f"edit_season_{outfit_id}"
            )
        
        # Display items with remove option
        st.markdown("#### Items in this outfit")
        st.caption("Click ✕ to remove an item")
        
        # Filter to only show items that are still in the outfit
        visible_items = [item for item in current_items if item.get("item_id") in items_to_keep]
        
        if visible_items:
            num_cols = min(len(visible_items), 4)
            cols = st.columns(num_cols)
            for j, item in enumerate(visible_items):
                with cols[j % num_cols]:
                    item_id = item.get("item_id")
                    
                    # Get image from session state
                    session_item = next(
                        (it for it in st.session_state.uploaded_items if it.get("item_id") == item_id),
                        None
                    )
                    img_src = ""
                    if session_item and session_item.get("image_base64"):
                        img_src = f"data:image/jpeg;base64,{session_item.get('image_base64')}"
                    elif item.get("image_url") and not str(item.get("image_url", "")).startswith("s3://"):
                        img_src = item.get("image_url")
                    
                    if img_src:
                        st.image(img_src, use_container_width=True)
                    
                    st.caption(f"{item.get('category', 'Item')}")
                    
                    # Remove button
                    if st.button("✕", key=f"remove_edit_{outfit_id}_{item_id}", use_container_width=True):
                        if item_id in st.session_state[f"edit_items_{outfit_id}"]:
                            st.session_state[f"edit_items_{outfit_id}"].remove(item_id)
                        st.rerun()
        else:
            st.warning("No items remaining. Add at least one item to save.")
        
        # Add from Wardrobe section
        with st.expander("Add from Wardrobe", expanded=False):
            st.caption("Select an item to add to this outfit")
            
            # Get categories already in outfit
            categories_in_outfit = set(
                item.get("category") for item in current_items 
                if item.get("item_id") in items_to_keep
            )
            
            # Filter to show items not already in outfit
            available_items = [
                item for item in st.session_state.uploaded_items 
                if item.get("item_id") not in items_to_keep
            ]
            
            if available_items:
                # Group by category
                categories = sorted(set(item.get("category", "Other") for item in available_items))
                selected_cat_filter = st.selectbox("Filter by Category", ["All"] + categories, key=f"edit_add_cat_filter_{outfit_id}")
                
                if selected_cat_filter != "All":
                    available_items = [item for item in available_items if item.get("category") == selected_cat_filter]
                
                # Display items in grid
                if available_items:
                    add_cols = st.columns(4)
                    for idx, wardrobe_item in enumerate(available_items):
                        with add_cols[idx % 4]:
                            # Get image
                            w_img_src = ""
                            if wardrobe_item.get("image_base64"):
                                w_img_src = f"data:image/jpeg;base64,{wardrobe_item.get('image_base64')}"
                            elif wardrobe_item.get("image_url") and not str(wardrobe_item.get("image_url", "")).startswith("s3://"):
                                w_img_src = wardrobe_item.get("image_url")
                            
                            if w_img_src:
                                st.image(w_img_src, use_container_width=True)
                            st.caption(f"{wardrobe_item.get('category', '')} - {wardrobe_item.get('subcategory', '') or wardrobe_item.get('brand', '')}")
                            
                            if st.button("Add", key=f"edit_add_item_{outfit_id}_{wardrobe_item.get('item_id')}", use_container_width=True, type="primary"):
                                # Add item to the outfit
                                st.session_state[f"edit_items_{outfit_id}"].append(wardrobe_item.get("item_id"))
                                # Also add to current_items for display
                                current_items.append(wardrobe_item)
                                st.rerun()
            else:
                st.info("All your wardrobe items are already in this outfit.")
        
        st.markdown("---")
        
        # Save and Cancel buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Save Changes", use_container_width=True, type="primary"):
                if len(items_to_keep) == 0:
                    st.error("Cannot save an outfit with no items!")
                else:
                    try:
                        api_client.update_outfit(
                            outfit_id=outfit_id,
                            name=new_name,
                            occasion=new_occasion,
                            season=new_season,
                            items=items_to_keep
                        )
                        # Clean up session state
                        if f"edit_items_{outfit_id}" in st.session_state:
                            del st.session_state[f"edit_items_{outfit_id}"]
                        get_cached_saved_outfits.clear()
                        st.success("Outfit updated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update: {e}")
        with btn_col2:
            if st.button("Cancel", use_container_width=True, type="primary"):
                # Clean up session state
                if f"edit_items_{outfit_id}" in st.session_state:
                    del st.session_state[f"edit_items_{outfit_id}"]
                st.rerun()
    
    try:
        with st.spinner("Loading saved outfits..."):
            response = get_cached_saved_outfits()
            saved_outfits = response.get("saved_outfits", [])
        
        if len(saved_outfits) == 0:
            st.info("You haven't saved any outfits yet. Go to Outfit Builder to create one!")
        else:
            # Filter dropdowns for occasion and season
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                occasion_filter = st.selectbox(
                    "Filter by Occasion",
                    ["All Occasions", "Casual", "Formal", "Business", "Party", "Everyday"],
                    key="saved_outfit_occasion_filter"
                )
            with filter_col2:
                season_filter = st.selectbox(
                    "Filter by Season",
                    ["All Seasons", "Spring", "Summer", "Fall", "Winter", "All-Season"],
                    key="saved_outfit_season_filter"
                )
            
            # Apply filters
            filtered_outfits = saved_outfits
            if occasion_filter != "All Occasions":
                filtered_outfits = [o for o in filtered_outfits if o.get("occasion") == occasion_filter]
            if season_filter != "All Seasons":
                filtered_outfits = [o for o in filtered_outfits if o.get("season") == season_filter]
            
            if not filtered_outfits:
                st.info(f"No outfits match the selected filters.")
            
            for i, outfit in enumerate(filtered_outfits):
                outfit_id = outfit.get("outfit_id", f"outfit_{i}")
                outfit_name = outfit.get('name') or f"Outfit {i+1}"
                
                # Header with title and buttons inline
                header_cols = st.columns([6, 1, 1], gap="small")
                with header_cols[0]:
                    st.markdown(f"### {outfit_name}")
                with header_cols[1]:
                    edit_clicked = st.button("Edit", key=f"edit_{outfit_id}", use_container_width=True)
                with header_cols[2]:
                    delete_clicked = st.button("Delete", key=f"del_{outfit_id}", use_container_width=True)
                
                # Tags below title
                st.markdown(
                    f'<span style="background:#E8D5E3; color:#6B4C98; padding:4px 10px; border-radius:12px; margin-right:8px;">{outfit.get("occasion", "Unknown")}</span>'
                    f'<span style="background:#f5f5f5; color:#666; padding:4px 10px; border-radius:12px;">{outfit.get("season", "Unknown")}</span>',
                    unsafe_allow_html=True
                )
                
                if edit_clicked:
                    edit_outfit_dialog(outfit)
                if delete_clicked:
                    try:
                        api_client.delete_outfit(outfit_id)
                        get_cached_saved_outfits.clear()
                        st.success("Deleted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
                
                items = outfit.get("items", [])
                if items:
                    # Limit to max 4 columns
                    num_cols = min(len(items), 4)
                    cols = st.columns(num_cols)
                    for j, item in enumerate(items[:4]):
                        with cols[j]:
                            # Try to get image from session state (has base64) or from backend URL
                            img_src = ""
                            item_id = item.get("item_id")
                            
                            # Check session state first for base64 image
                            session_item = next(
                                (it for it in st.session_state.uploaded_items if it.get("item_id") == item_id),
                                None
                            )
                            if session_item and session_item.get("image_base64"):
                                img_src = f"data:image/jpeg;base64,{session_item.get('image_base64')}"
                            elif item.get("image_url") and not str(item.get("image_url", "")).startswith("s3://"):
                                img_src = item.get("image_url")
                            
                            if img_src:
                                st.markdown(
                                    f'<div style="border-radius:12px; overflow:hidden; border:1px solid #E8D5E3; background:white;">'
                                    f'<img src="{img_src}" style="width:100%; height:150px; object-fit:cover;" />'
                                    f'<div style="padding:8px; text-align:center; font-size:12px; color:#2E2E2E;">{item.get("category", "Item")}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f'<div style="border-radius:12px; border:1px solid #E8D5E3; background:#f5f5f5; height:150px; display:flex; align-items:center; justify-content:center; color:#999;">'
                                    f'{item.get("category", "Item")}</div>',
                                    unsafe_allow_html=True
                                )
                
                # Feedback section for saved outfit
                saved_outfit_key = f"saved_{outfit_id}"
                saved_feedback = st.session_state.outfit_feedback.get(saved_outfit_key, {})
                
                with st.expander("💬 Rate & Review This Outfit"):
                    feedback_col1, feedback_col2 = st.columns([1, 2])
                    with feedback_col1:
                        saved_rating = st.select_slider(
                            "Your Rating",
                            options=[1, 2, 3, 4, 5],
                            value=saved_feedback.get("rating", 3),
                            key=f"rating_{saved_outfit_key}"
                        )
                        st.caption("⭐ 1 = Poor, ⭐⭐⭐⭐⭐ 5 = Excellent")
                    
                    with feedback_col2:
                        saved_comment = st.text_area(
                            "Your Feedback",
                            value=saved_feedback.get("comment", ""),
                            placeholder="How did this outfit work for you?",
                            key=f"comment_{saved_outfit_key}",
                            height=100
                        )
                    
                    if st.button(" Submit Feedback", key=f"submit_{saved_outfit_key}"):
                        st.session_state.outfit_feedback[saved_outfit_key] = {
                            "rating": saved_rating,
                            "comment": saved_comment,
                            "occasion": outfit.get('occasion'),
                            "season": outfit.get('season')
                        }
                        st.success("Feedback saved!")
                        st.rerun()
                    
                    # Display existing feedback
                    if saved_feedback.get("rating") or saved_feedback.get("comment"):
                        st.info(f"**Your feedback:** ⭐ {saved_feedback.get('rating', 'N/A')}/5 - {saved_feedback.get('comment', 'No comment')}")
                
                st.markdown("---")
    
    except Exception as e:
        st.error(f"Failed to load saved outfits: {str(e)}")