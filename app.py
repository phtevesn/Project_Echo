"""
Project Echo - Socio-Economic Income Predictor
GUI built with tkinter; ensemble model (XGBoost + LightGBM + CatBoost)
loaded via joblib from income_pipeline.pkl (ACS/PUMS dataset).
"""

import os
import sys
import math
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "income_pipeline.pkl")

# COW - Class of Worker (ACS codes 1-9)
COW_OPTIONS = [
    (1, "Private for-profit company"),
    (2, "Private not-for-profit"),
    (3, "Local government"),
    (4, "State government"),
    (5, "Federal government"),
    (6, "Self-employed (incorporated)"),
    (7, "Self-employed (not incorporated)"),
    (8, "Family business (unpaid)"),
    (9, "Unemployed / Never worked"),
]

# SCHL - Education level (grouped: 1-15->1.0 <HS, 16-17->2.0 HS/GED, 18-19->3.0 Some College,
#         20->4.0 Associate, 21->5.0 Bachelor, 22-24->6.0 Graduate+)
SCHL_OPTIONS = [
    (1.0, "Less than High School"),
    (2.0, "High School Diploma / GED"),
    (3.0, "Some College (no degree)"),
    (4.0, "Associate's Degree"),
    (5.0, "Bachelor's Degree"),
    (6.0, "Graduate Degree"),
]

# MAR - Marital Status (ACS codes 1-5)
MAR_OPTIONS = [
    (1, "Married"),
    (2, "Widowed"),
    (3, "Divorced"),
    (4, "Separated"),
    (5, "Never married"),
]

# SEX (ACS: 1=Male, 2=Female)
SEX_OPTIONS = [(1, "Male"), (2, "Female")]

# OCCP - Occupation group codes used by the pipeline
OCCP_OPTIONS = [
    ("MGR",   "Management"),
    ("BUS",   "Business & Financial Operations"),
    ("FIN",   "Financial Specialists"),
    ("CMM",   "Computer & Mathematical"),
    ("ENG",   "Architecture & Engineering"),
    ("SCI",   "Life, Physical & Social Sciences"),
    ("CMS",   "Community & Social Services"),
    ("LGL",   "Legal"),
    ("EDU",   "Education, Training & Library"),
    ("ENT",   "Arts, Design, Entertainment & Media"),
    ("MED",   "Healthcare Practitioners & Technical"),
    ("HLS",   "Healthcare Support"),
    ("PRT",   "Protective Service"),
    ("EAT",   "Food Preparation & Serving"),
    ("CLN",   "Building & Grounds Cleaning"),
    ("PRS",   "Personal Care & Service"),
    ("SAL",   "Sales & Related"),
    ("OFF",   "Office & Administrative Support"),
    ("FFF",   "Farming, Fishing & Forestry"),
    ("CON",   "Construction & Extraction"),
    ("EXT",   "Extraction Workers"),
    ("RPR",   "Installation, Maintenance & Repair"),
    ("PRD",   "Production"),
    ("TRN",   "Transportation & Material Moving"),
    ("MIL",   "Military Specific"),
    ("UNEMP", "Unemployed / Not working"),
]

# RELP_GROUP - Relationship categories
RELP_OPTIONS = [
    ("reference",       "Reference person (head of household)"),
    ("spouse",          "Spouse"),
    ("child_dep",       "Child / Dependent"),
    ("parent",          "Parent"),
    ("extended_family", "Extended family member"),
    ("non_family",      "Non-family household member"),
    ("other",           "Other relationship"),
    ("group_quarters",  "Group quarters"),
    ("unknown",         "Unknown"),
]

# RAC1P_GROUP - Race categories
RACE_OPTIONS = [
    ("white",      "White"),
    ("black",      "Black or African American"),
    ("asian",      "Asian"),
    ("indigenous", "American Indian / Alaska Native"),
    ("pacific",    "Native Hawaiian / Pacific Islander"),
    ("multi",      "Two or more races"),
    ("other",      "Some other race"),
]

# POBP - Place of Birth (US state FIPS codes + foreign region groups)
POBP_OPTIONS = [
    (6.0,   "California"),
    (36.0,  "New York"),
    (48.0,  "Texas"),
    (12.0,  "Florida"),
    (17.0,  "Illinois"),
    (42.0,  "Pennsylvania"),
    (39.0,  "Ohio"),
    (26.0,  "Michigan"),
    (37.0,  "North Carolina"),
    (13.0,  "Georgia"),
    (34.0,  "New Jersey"),
    (53.0,  "Washington"),
    (25.0,  "Massachusetts"),
    (47.0,  "Tennessee"),
    (27.0,  "Minnesota"),
    (51.0,  "Virginia"),
    (55.0,  "Wisconsin"),
    (29.0,  "Missouri"),
    (18.0,  "Indiana"),
    (24.0,  "Maryland"),
    (45.0,  "South Carolina"),
    (40.0,  "Oklahoma"),
    (21.0,  "Kentucky"),
    (22.0,  "Louisiana"),
    (41.0,  "Oregon"),
    (9.0,   "Connecticut"),
    (35.0,  "New Mexico"),
    (19.0,  "Iowa"),
    (5.0,   "Arkansas"),
    (49.0,  "Utah"),
    (32.0,  "Nevada"),
    (20.0,  "Kansas"),
    (28.0,  "Mississippi"),
    (4.0,   "Arizona"),
    (8.0,   "Colorado"),
    (23.0,  "Maine"),
    (1.0,   "Alabama"),
    (31.0,  "Nebraska"),
    (54.0,  "West Virginia"),
    (16.0,  "Idaho"),
    (33.0,  "New Hampshire"),
    (30.0,  "Montana"),
    (38.0,  "North Dakota"),
    (46.0,  "South Dakota"),
    (50.0,  "Vermont"),
    (2.0,   "Alaska"),
    (56.0,  "Wyoming"),
    (15.0,  "Hawaii"),
    (10.0,  "Delaware"),
    (11.0,  "District of Columbia"),
    (44.0,  "Rhode Island"),
    (100.0, "Puerto Rico"),
    (101.0, "US Island Areas"),
    (102.0, "Born abroad (US parents)"),
    (103.0, "Foreign-born (naturalized)"),
    (104.0, "Mexico"),
    (105.0, "Central / South America"),
    (106.0, "Other foreign country"),
]


def _row(parent, label_text, widget, row, col_offset=0):
    tk.Label(parent, text=label_text, anchor="w").grid(
        row=row, column=col_offset, sticky="w", padx=8, pady=4
    )
    widget.grid(row=row, column=col_offset + 1, sticky="ew", padx=8, pady=4)


def _combobox(parent, options, width=None):
    labels = [lbl for _, lbl in options] if isinstance(options[0], tuple) else options
    if width is None:
        width = max(len(lbl) for lbl in labels) + 2
    cb = ttk.Combobox(parent, values=labels, state="readonly", width=width)
    cb.current(0)
    return cb


def _spinbox(parent, from_, to, default=0, width=12):
    var = tk.StringVar(value=str(default))
    sb = ttk.Spinbox(parent, from_=from_, to=to, textvariable=var, width=width)
    return sb, var


def preprocess(raw: dict, pipeline: dict) -> pd.DataFrame:
    """Convert raw GUI inputs into the feature DataFrame expected by the models."""
    relp_enc  = pipeline["relp_ordinal_encoder"]
    race_enc  = pipeline["race_ordinal_encoder"]
    occp_enc  = pipeline["occp_ordinal_encoder"]
    pobp_map  = pipeline["pobp_frequency_map"]
    feat_cols = pipeline["feature_columns"]

    relp_val = relp_enc.transform(pd.DataFrame([[raw["RELP_GROUP"]]], columns=["RELP_GROUP"]))[0][0]
    race_val = race_enc.transform(pd.DataFrame([[raw["RAC1P_GROUP"]]], columns=["RAC1P_GROUP"]))[0][0]
    occp_val = occp_enc.transform(pd.DataFrame([[raw["OCCP_GROUP"]]], columns=["OCCP_GROUP"]))[0][0]

    occp_ohe = {f"OCCP_GROUP_{float(i):.1f}": (1 if i == occp_val else 0) for i in range(25)}
    mar_ohe  = {f"MAR_{float(i):.1f}": (1 if i == raw["MAR"] else 0) for i in range(1, 6)}
    relp_ohe = {f"RELP_GROUP_{float(i):.1f}": (1 if i == relp_val else 0) for i in [0, 1, 3, 4, 5, 6, 7, 8]}
    rac_ohe  = {f"RAC1P_GROUP_{float(i):.1f}": (1 if i == race_val else 0) for i in range(7)}

    row = {"AGEP": raw["AGEP"], "COW": raw["COW"], "SCHL": raw["SCHL"],
           "WKHP": raw["WKHP"], "SEX": raw["SEX"]}
    row.update(occp_ohe)
    row.update(mar_ohe)
    row.update(relp_ohe)
    row.update(rac_ohe)
    row["POBP_freq"] = pobp_map.get(raw["POBP"], pobp_map.median())

    return pd.DataFrame([row])[feat_cols]


class EchoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Project Echo: Socio-Economic Income Predictor")
        self.resizable(True, False)
        self._load_model()
        self._build_ui()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showwarning(
                "Model Not Found",
                f"income_pipeline.pkl was not found at:\n{MODEL_PATH}\n\n"
                "Place the file next to app.py (or the .exe) and restart."
            )
            self.pipeline = None
        else:
            try:
                self.pipeline = joblib.load(MODEL_PATH)
            except Exception as exc:
                messagebox.showerror("Load Error", f"Could not load model:\n{exc}")
                self.pipeline = None

    def _build_ui(self):
        header = tk.Frame(self, bg="#1a1a2e")
        header.pack(fill="x")
        tk.Label(
            header, text="Project Echo", font=("Helvetica", 18, "bold"),
            fg="#e94560", bg="#1a1a2e", pady=10
        ).pack()
        tk.Label(
            header, text="Socio-Economic Income Predictor",
            font=("Helvetica", 10), fg="#a8a8b3", bg="#1a1a2e", pady=2
        ).pack()

        form = tk.LabelFrame(self, text=" Personal Information ", padx=6, pady=6)
        form.pack(padx=16, pady=10, fill="both")

        self._age_sb,   self._age_var   = _spinbox(form, 16, 95, 30)
        self._hours_sb, self._hours_var = _spinbox(form, 1,  99, 40)
        _row(form, "Age",                 self._age_sb,   0)
        _row(form, "Hours Worked / Week", self._hours_sb, 1)

        self._cow_cb  = _combobox(form, COW_OPTIONS)
        self._schl_cb = _combobox(form, SCHL_OPTIONS)
        self._mar_cb  = _combobox(form, MAR_OPTIONS)
        self._sex_cb  = _combobox(form, SEX_OPTIONS)
        self._occp_cb = _combobox(form, OCCP_OPTIONS)
        self._relp_cb = _combobox(form, RELP_OPTIONS)
        self._race_cb = _combobox(form, RACE_OPTIONS)
        self._pobp_cb = _combobox(form, POBP_OPTIONS)

        _row(form, "Class of Worker", self._cow_cb,  0, col_offset=2)
        _row(form, "Education Level", self._schl_cb, 1, col_offset=2)
        _row(form, "Marital Status",  self._mar_cb,  2, col_offset=2)
        _row(form, "Sex",             self._sex_cb,  3, col_offset=2)
        _row(form, "Occupation",      self._occp_cb, 4, col_offset=2)
        _row(form, "Household Role",  self._relp_cb, 5, col_offset=2)
        _row(form, "Race",            self._race_cb, 6, col_offset=2)
        _row(form, "Place of Birth",  self._pobp_cb, 7, col_offset=2)

        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=(4, 8))
        ttk.Button(btn_frame, text="Predict Estimated Income", command=self._predict).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Reset", command=self._reset).pack(side="left", padx=6)

        result_frame = tk.LabelFrame(self, text=" Prediction Result ", padx=8, pady=8)
        result_frame.pack(padx=16, pady=(0, 16), fill="x")

        self._result_var = tk.StringVar(value="-")
        self._result_label = tk.Label(
            result_frame, textvariable=self._result_var,
            font=("Helvetica", 16, "bold"), fg="#2d6a4f"
        )
        self._result_label.pack()

        self._detail_var = tk.StringVar(value="")
        tk.Label(result_frame, textvariable=self._detail_var, font=("Helvetica", 9), fg="#555").pack()

    def _collect_raw(self) -> dict:
        try:
            age = int(self._age_var.get())
        except ValueError:
            raise ValueError("Age must be a whole number between 16 and 95.")
        try:
            hours = int(self._hours_var.get())
        except ValueError:
            raise ValueError("Hours worked per week must be a whole number between 1 and 99.")

        if not (16 <= age <= 95):
            raise ValueError(f"Age must be between 16 and 95 (got {age}).")
        if not (1 <= hours <= 99):
            raise ValueError(f"Hours worked per week must be between 1 and 99 (got {hours}).")

        return {
            "AGEP":        age,
            "COW":         COW_OPTIONS[self._cow_cb.current()][0],
            "SCHL":        SCHL_OPTIONS[self._schl_cb.current()][0],
            "WKHP":        hours,
            "SEX":         SEX_OPTIONS[self._sex_cb.current()][0],
            "OCCP_GROUP":  OCCP_OPTIONS[self._occp_cb.current()][0],
            "MAR":         MAR_OPTIONS[self._mar_cb.current()][0],
            "RELP_GROUP":  RELP_OPTIONS[self._relp_cb.current()][0],
            "RAC1P_GROUP": RACE_OPTIONS[self._race_cb.current()][0],
            "POBP":        POBP_OPTIONS[self._pobp_cb.current()][0],
        }

    def _predict(self):
        if self.pipeline is None:
            messagebox.showerror("No Model", "income_pipeline.pkl is not loaded.")
            return

        try:
            raw = self._collect_raw()
            X = preprocess(raw, self.pipeline)
            preds = [float(m.predict(X)[0]) for m in self.pipeline["models"].values()]
            income_usd = math.exp(sum(preds) / len(preds))
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))
            return

        if income_usd >= 200_000:
            color, bracket = "#1b4332", "High Income"
        elif income_usd >= 100_000:
            color, bracket = "#2d6a4f", "Upper-Middle Income"
        elif income_usd >= 50_000:
            color, bracket = "#457b9d", "Middle Income"
        elif income_usd >= 25_000:
            color, bracket = "#e07b00", "Lower-Middle Income"
        else:
            color, bracket = "#c0392b", "Lower Income"

        self._result_var.set(f"Estimated Annual Income:  ${income_usd:,.0f}  ({bracket})")
        self._result_label.config(fg=color)

        model_detail = "  |  ".join(
            f"{n.upper()}: ${math.exp(p):,.0f}"
            for n, p in zip(self.pipeline["models"].keys(), preds)
        )
        self._detail_var.set(f"Model breakdown  ->  {model_detail}")

    def _reset(self):
        self._age_var.set("30")
        self._hours_var.set("40")
        for cb in (self._cow_cb, self._schl_cb, self._mar_cb, self._sex_cb,
                   self._occp_cb, self._relp_cb, self._race_cb, self._pobp_cb):
            cb.current(0)
        self._result_var.set("-")
        self._result_label.config(fg="#2d6a4f")
        self._detail_var.set("")


if __name__ == "__main__":
    app = EchoApp()
    app.mainloop()
