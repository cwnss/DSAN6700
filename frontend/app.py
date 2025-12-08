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

/* Main background + text */
.stApp { background:#FFFCF5; color:#2E2E2E; }

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
.main .block-container{ background:transparent !important; }

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
  font-size: 15px !important;
  height: 36px !important;
  width: 100% !important;
  min-width: 80px !important;
  text-align: center !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: background-color 0.2s ease !important;
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

# Backend connection status
try:
    api_client.health_check()
    st.sidebar.success("✓ Backend connected")
except Exception as e:
    st.sidebar.error(f"✗ Backend offline")

if "page" not in st.session_state:
    st.session_state.page = "My Wardrobe"

# --- Sidebar buttons ---
if st.sidebar.button("My Wardrobe", use_container_width=True):
    st.session_state.page = "My Wardrobe"
if st.sidebar.button("Outfit Builder", use_container_width=True):
    st.session_state.page = "Outfit Builder"
if st.sidebar.button("Saved Outfits", use_container_width=True):
    st.session_state.page = "Saved Outfits"

page = st.session_state.page

# Add JavaScript to highlight active page button
current_page = st.session_state.page
st.sidebar.markdown(
    f"""
    <script>
    (function() {{
        // Wait for DOM to be ready
        setTimeout(function() {{
            var buttons = document.querySelectorAll('section[data-testid="stSidebar"] .stButton > button');
            var pageMap = {{
                'My Wardrobe': 'My Wardrobe',
                'Outfit Builder': 'Outfit Builder',
                'Saved Outfits': 'Saved Outfits'
            }};
            var currentPage = '{current_page}';
            
            buttons.forEach(function(button) {{
                var buttonText = button.textContent.trim();
                if (buttonText === currentPage) {{
                    button.classList.add('active-page');
                }} else {{
                    button.classList.remove('active-page');
                }}
            }});
        }}, 100);
    }})();
    </script>
    """,
    unsafe_allow_html=True
)

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

# --- My Wardrobe Page ---
if page == "My Wardrobe":
    # Load items from backend on first load (only once per session)
    if not st.session_state.items_loaded_from_backend:
        try:
            response = get_cached_wardrobe_items()
            backend_items = response.get("items", [])
            
            # Convert backend items to match session state format
            # Backend items have image_url, but we need to keep them as-is
            # Newly uploaded items will have image_base64 for immediate display
            for backend_item in backend_items:
                item_id = backend_item.get("item_id")
                # Check if item already exists in session state (by item_id)
                existing_item = None
                for existing in st.session_state.uploaded_items:
                    if existing.get("item_id") == item_id:
                        existing_item = existing
                        break
                
                if existing_item:
                    # Merge: Keep tags from session state, update image_url from backend
                    existing_item["image_url"] = backend_item.get("image_url", existing_item.get("image_url"))
                    # Preserve all tags from session state (they may have been edited)
                    # Don't overwrite tags with backend data
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
    # Fix pluralization: "1 item" vs "X items"
    item_count = len(st.session_state.uploaded_items) if st.session_state.uploaded_items else 0
    if item_count == 1:
        st.write(f"{item_count} item in your collection")
    else:
        st.write(f"{item_count} items in your collection")

    col_add = st.columns([5, 1])[1]
    with col_add:
        if st.button("+ Add New Item"):
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
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Upload Your First Item!"):
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
                "Season",
                ["Spring", "Summer", "Fall", "Winter", "All-Season"]
            )
            
            brand = st.text_input("Brand", placeholder="e.g., Christopher Esber")
            colors = st.text_input(
                "Colors (comma-separated)", placeholder="e.g., brown, beige"
            )
            
            occasions = st.multiselect(
                "Occasions",
                ["Casual", "Formal", "Business", "Athletic", "Party", "Everyday"],
            )
            notes = st.text_area(
                "Notes", placeholder="Add any personal notes about this item"
            )
            
            # Add required field indicator at bottom
            st.markdown("<p style='color: #6B4C98; font-size: 0.875rem;'>* Required fields</p>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add to Wardrobe"):
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
                            
                            # Upload to backend
                            response = api_client.upload_wardrobe_item(
                                image_file=file_bytes_io,
                                filename=uploaded_file.name,
                                category=category
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
                                "season": season,
                                "subcategory": subcategory,
                                "brand": brand,
                                "colors": [c.strip() for c in colors.split(",") if c.strip()] if colors else [],
                                "occasions": occasions,
                                "notes": notes,
                                "image_base64": image_base64  # Store as base64 string
                            }
                            
                            if "uploaded_items" not in st.session_state:
                                st.session_state.uploaded_items = []
                            st.session_state.uploaded_items.append(item_data)
                            
                            # Clear cache so backend items refresh
                            get_cached_wardrobe_items.clear()
                            
                            st.session_state.show_uploader = False
                            st.success(f"Added '{brand or 'New Item'}' to your wardrobe! (ID: {response.get('item_id')})")
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Failed to upload item: {str(e)}")
            with col2:
                if st.button("Cancel"):
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
        
        # Get unique values for filters
        all_categories = ["All Categories"] + sorted(list(set([item.get("category", "") for item in st.session_state.uploaded_items if item.get("category")])))
        all_occasions = ["All Occasions"] + sorted(list(set([occ for item in st.session_state.uploaded_items for occ in item.get("occasions", [])])))
        all_colors = ["All Colors"] + sorted(list(set([color for item in st.session_state.uploaded_items for color in item.get("colors", [])])))
        
        # Filter dropdowns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.filter_category = st.selectbox(
                "Filter by Category",
                all_categories,
                index=all_categories.index(st.session_state.filter_category) if st.session_state.filter_category in all_categories else 0
            )
        with col2:
            st.session_state.filter_occasion = st.selectbox(
                "Filter by Occasion",
                all_occasions,
                index=all_occasions.index(st.session_state.filter_occasion) if st.session_state.filter_occasion in all_occasions else 0
            )
        with col3:
            st.session_state.filter_color = st.selectbox(
                "Filter by Color",
                all_colors,
                index=all_colors.index(st.session_state.filter_color) if st.session_state.filter_color in all_colors else 0
            )
        
        # Filter items - only apply filter if not "All" option
        filtered_items = st.session_state.uploaded_items.copy()
        
        # Category filter
        filter_cat = str(st.session_state.filter_category).strip() if st.session_state.filter_category else ""
        if filter_cat and filter_cat != "All Categories":
            filtered_items = [item for item in filtered_items if item.get("category") == filter_cat]
        
        # Occasion filter
        filter_occ = str(st.session_state.filter_occasion).strip() if st.session_state.filter_occasion else ""
        if filter_occ and filter_occ != "All Occasions":
            filtered_items = [item for item in filtered_items if filter_occ in item.get("occasions", [])]
        
        # Color filter - explicitly skip if "All Colors" is selected
        filter_color = str(st.session_state.filter_color).strip() if st.session_state.filter_color else ""
        if filter_color and filter_color != "All Colors":
            # Only filter by color if NOT "All Colors"
            filtered_items = [item for item in filtered_items if filter_color.lower() in [c.lower() for c in item.get("colors", [])]]
        # If "All Colors" is selected, don't filter by color (show all items regardless of color)
        
        # Display filtered items in card layout using st.columns for grid
        if filtered_items:
            # Create grid: 4 columns per row
            items_per_row = 4
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
                            f'<div style="display:flex; flex-wrap:wrap; align-items:center; padding:0 12px 8px 12px;">{all_tags}</div>'
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
                                    "Season",
                                    ["Spring", "Summer", "Fall", "Winter", "All-Season"],
                                    default=season_value,
                                    key=f"edit_season_{item_id}"
                                )
                                edit_brand = st.text_input("Brand", value=item.get("brand", ""), key=f"edit_brand_{item_id}")
                                colors_str = ", ".join(item.get("colors", [])) if item.get("colors") else ""
                                edit_colors = st.text_input("Colors (comma-separated)", value=colors_str, key=f"edit_colors_{item_id}")
                                edit_occasions = st.multiselect(
                                    "Occasions",
                                    ["Casual", "Formal", "Business", "Athletic", "Party", "Everyday"],
                                    default=item.get("occasions", []),
                                    key=f"edit_occ_{item_id}"
                                )
                                edit_notes = st.text_area("Notes", value=item.get("notes", ""), key=f"edit_notes_{item_id}", height=60)
                                
                                if st.button("Save Changes", key=f"save_{item_id}", use_container_width=True):
                                    # Find and update the actual item in session state (not the filtered copy)
                                    for session_item in st.session_state.uploaded_items:
                                        if session_item.get("item_id") == item_id:
                                            # Update all tags in the actual session state item
                                            session_item["category"] = edit_category
                                            session_item["subcategory"] = edit_subcategory
                                            session_item["season"] = edit_season
                                            session_item["brand"] = edit_brand
                                            session_item["colors"] = [c.strip() for c in edit_colors.split(",") if c.strip()] if edit_colors else []
                                            session_item["occasions"] = edit_occasions
                                            session_item["notes"] = edit_notes
                                            break
                                    st.success("✓ Changes saved!")
                                    st.rerun()
                        
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
    st.write("Create an outfit by selecting items from each category.")

    if len(st.session_state.uploaded_items) == 0:
        st.info("Upload some items in 'My Wardrobe' first!")
    else:
        # Initialize outfit selections in session state
        if "outfit_selections" not in st.session_state:
            st.session_state.outfit_selections = {}
        
        # Group items by category
        categories = ["Tops", "Bottoms", "Dresses", "Outerwear", "Shoes", "Accessories"]
        items_by_category = {cat: [] for cat in categories}
        
        for item in st.session_state.uploaded_items:
            cat = item.get("category", "Other")
            if cat in items_by_category:
                items_by_category[cat].append(item)
        
        # Helper function to get image source
        def get_image_src(item):
            if item.get("image_base64"):
                return f"data:image/jpeg;base64,{item.get('image_base64')}"
            elif item.get("image_url") and not item.get("image_url", "").startswith("s3://"):
                return item.get("image_url")
            return ""
        
        # Display each category with selectable items
        for category in categories:
            items = items_by_category.get(category, [])
            if not items:
                continue
            
            st.markdown(f"#### {category}")
            
            cols = st.columns(min(len(items), 4))
            for idx, item in enumerate(items[:4]):  # Show max 4 per category
                with cols[idx]:
                    item_id = item.get("item_id")
                    is_selected = st.session_state.outfit_selections.get(category) == item_id
                    border = "3px solid #6B4C98" if is_selected else "1px solid #E8D5E3"
                    
                    img_src = get_image_src(item)
                    item_name = item.get("subcategory") or item.get("brand") or category
                    
                    if img_src:
                        st.markdown(
                            f'<div style="border-radius:12px; overflow:hidden; border:{border}; background:white; margin-bottom:8px;">'
                            f'<img src="{img_src}" style="width:100%; height:120px; object-fit:cover;" />'
                            f'<div style="padding:6px; text-align:center; font-size:12px; color:#2E2E2E;">{item_name}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div style="border-radius:12px; border:{border}; background:#f5f5f5; height:150px; display:flex; align-items:center; justify-content:center; color:#999;">'
                            f'No image</div>',
                            unsafe_allow_html=True
                        )
                    
                    btn_label = "✓ Selected" if is_selected else "Select"
                    if st.button(btn_label, key=f"outfit_select_{category}_{item_id}", use_container_width=True):
                        if is_selected:
                            # Deselect
                            del st.session_state.outfit_selections[category]
                        else:
                            # Select
                            st.session_state.outfit_selections[category] = item_id
                        st.rerun()
        
        st.markdown("---")
        
        # Show outfit preview
        selected_items = []
        for cat, item_id in st.session_state.outfit_selections.items():
            item = next((i for i in st.session_state.uploaded_items if i.get("item_id") == item_id), None)
            if item:
                selected_items.append(item)
        
        if selected_items:
            st.markdown("### Your Outfit")
            
            preview_cols = st.columns(len(selected_items))
            for idx, item in enumerate(selected_items):
                with preview_cols[idx]:
                    img_src = get_image_src(item)
                    if img_src:
                        st.image(img_src, use_container_width=True)
                    st.caption(f"{item.get('category')} - {item.get('subcategory', '')}")
            
            # Outfit name and details for saving
            st.markdown("#### Save This Outfit")
            outfit_name = st.text_input("Outfit Name", placeholder="e.g., Casual Friday, Date Night, Work Meeting", key="outfit_name")
            save_col1, save_col2 = st.columns(2)
            with save_col1:
                save_occasion = st.selectbox("Occasion", ["Casual", "Formal", "Business", "Party", "Everyday"], key="save_occasion")
            with save_col2:
                save_season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter", "All-Season"], key="save_season")
            
            if st.button("Save Outfit", use_container_width=True):
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
            
            if st.button("Clear Selections", use_container_width=True):
                st.session_state.outfit_selections = {}
                st.rerun()
        else:
            st.info("Select items from the categories above to build your outfit.")

# --- Saved Outfits Page ---
elif page == "Saved Outfits":
    st.subheader("Saved Outfits")
    
    try:
        with st.spinner("Loading saved outfits..."):
            response = get_cached_saved_outfits()
            saved_outfits = response.get("saved_outfits", [])
        
        if len(saved_outfits) == 0:
            st.info("You haven't saved any outfits yet. Go to Outfit Builder to create one!")
        else:
            for i, outfit in enumerate(saved_outfits):
                outfit_id = outfit.get("outfit_id", f"outfit_{i}")
                outfit_name = outfit.get('name') or f"Outfit {i+1}"
                
                # Header row with title and delete button
                title_col, delete_col = st.columns([5, 1])
                with title_col:
                    st.markdown(f"### {outfit_name}")
                    st.markdown(
                        f'<span style="background:#E8D5E3; color:#6B4C98; padding:4px 10px; border-radius:12px; margin-right:8px;">{outfit.get("occasion", "Unknown")}</span>'
                        f'<span style="background:#f5f5f5; color:#666; padding:4px 10px; border-radius:12px;">{outfit.get("season", "Unknown")}</span>',
                        unsafe_allow_html=True
                    )
                with delete_col:
                    if st.button("Delete", key=f"del_{outfit_id}"):
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
                        st.success("Feedback saved! 💜")
                        st.rerun()
                    
                    # Display existing feedback
                    if saved_feedback.get("rating") or saved_feedback.get("comment"):
                        st.info(f"**Your feedback:** ⭐ {saved_feedback.get('rating', 'N/A')}/5 - {saved_feedback.get('comment', 'No comment')}")
                
                st.markdown("---")
    
    except Exception as e:
        st.error(f"Failed to load saved outfits: {str(e)}")
