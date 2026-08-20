"""
comando: modifica le 3 righe qui sotto se serve, poi lancia con:

    python JSONtoCalendar.py

Serve solo matplotlib (pip install matplotlib).
"""
import json
import os
import datetime
import matplotlib.pyplot as plt

INPUT_JSON = "outputs/schedule_final.json"      # file da leggere
OUTPUT_DIR = "outputs"           # cartella dove salvare l'immagine
DATA_GIORNO_0 = datetime.date(2026, 12, 7)  # a che data corrisponde il "giorno 0"

# --- Legge il JSON e prende lo schedule -----------------------------------
with open(INPUT_JSON, encoding="utf-8") as f:
    dati = json.load(f)
schedule = dati["schedule"] if "schedule" in dati else dati

# --- Costruisce righe/colonne della tabella ---------------------------------
nomi_turni = ["Mattina", "Pomeriggio", "Notte"]
giorni = sorted(schedule.keys(), key=int)

etichette_righe = []
righe = []
for g in giorni:
    data_giorno = DATA_GIORNO_0 + datetime.timedelta(days=int(g))
    etichette_righe.append(f"Giorno {g}\n{data_giorno.strftime('%d/%m/%Y')}")
    riga = []
    for turno in ["0", "1", "2"]:
        lavoratori = schedule[g].get(turno, [])
        riga.append(", ".join(str(w) for w in lavoratori) if lavoratori else "-")
    righe.append(riga)

# --- Disegna e salva la tabella ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, len(giorni) * 0.4 + 1))
ax.axis("off")
tabella = ax.table(
    cellText=righe,
    rowLabels=etichette_righe,
    colLabels=nomi_turni,
    loc="center",
    cellLoc="center",
)
tabella.auto_set_font_size(False)
tabella.set_fontsize(9)
tabella.scale(1, 1.8)

os.makedirs(OUTPUT_DIR, exist_ok=True)
percorso_output = os.path.join(OUTPUT_DIR, "tabella_turno.png")
plt.savefig(percorso_output, dpi=200, bbox_inches="tight")
print(f"Tabella salvata in: {percorso_output}")