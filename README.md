# SmartScheduler

Sistema multi-agente per la pianificazione automatica dei turni ospedalieri,
che combina LLM (via Groq) per generazione/raffinamento di codice OR-Tools
con verifica simbolica (senza LLM) dei vincoli rigidi e delle metriche di
equita'. Orchestrazione a grafo con **LangGraph**.

## Installazione

```bash
pip install -r requirements.txt
```

Poi apri `config.py` e inserisci una API key Groq valida in `GROQ_API_KEY`
(per istruzioni progetto, la chiave e il nome modello sono hardcoded nel
file di configurazione, non letti da variabili d'ambiente).

## Esecuzione

```bash
python main.py --input examples/input_case_a.txt
python main.py --input examples/input_case_b.txt
```

Gli output vengono scritti in `outputs/`:

- `schedule_final.json` — assegnazioni finali, stato, errori residui, punteggi di equita'
- `schedule_model.py` — l'ultimo modello OR-Tools generato (quello che ha prodotto lo schedule finale)
- `fairness_report.txt` — report leggibile delle metriche di equita' per lavoratore

## Struttura del progetto

```
smart_scheduler/
├── main.py                    # Entry point CLI
├── config.py                  # Configurazione (API key, soglie, orizzonte)
├── core/
│   ├── state.py                # TypedDict ScheduleState (memoria condivisa del grafo)
│   ├── models.py                # Worker, Shift, Schedule
│   ├── graph.py                  # Costruzione del grafo LangGraph e router condizionali
│   └── orchestrator.py            # Esecuzione end-to-end e scrittura output
├── agents/
│   ├── base.py                     # Interfaccia comune degli agenti
│   ├── preference_agent.py          # Fase 1: LLM, testo -> preferenze JSON
│   ├── drafting_agent.py             # Fase 2: LLM genera codice OR-Tools
│   ├── verification_agent.py          # Fase 3: SIMBOLICO, nessun LLM
│   └── refinement_agent.py             # Fase 4: LLM corregge il codice precedente
├── solver/
│   ├── or_tools_wrapper.py              # Esecuzione sandboxed del codice generato
│   ├── validator.py                      # Verifica di tutti i vincoli rigidi
│   └── fairness.py                        # Calcolo dei punteggi di soddisfazione
├── llm/
│   ├── groq_client.py                      # Client Groq con rate limiting + retry
│   ├── token_counter.py                     # TokenBucket + stima token
│   └── prompt_templates.py                   # Prompt dei tre agenti LLM
├── utils/
│   ├── file_parser.py                         # Parsing del file di input
│   ├── logger.py                               # Logging strutturato
│   └── exceptions.py                            # Eccezioni custom
├── tests/                                        # Unit + integration test (vedi sotto)
├── examples/                                      # Input di esempio per Caso A e Caso B
└── docs/relazione.md                               # Relazione tecnica del progetto
```

## Test

```bash
pytest tests/ -v
```

`test_validator.py` e `test_solver.py` sono puramente Python (nessuna
dipendenza esterna) e girano ovunque. `test_integration.py` esegue il
grafo LangGraph completo con un client Groq finto (deterministico, nessuna
chiamata di rete) ma con **OR-Tools reale**: si auto-salta se `ortools` o
`langgraph` non sono installati.

## Nota sullo sviluppo e validazione

Il progetto e' stato validato con run reali contro l'API Groq per
entrambi i casi d'uso (non solo test unitari) — vedi `docs/relazione.md`,
sezioni 5 e 6, per il diario delle correzioni emerse durante la
validazione reale e i risultati numerici ottenuti (gap di equita',
token consumati, tempi di esecuzione).

## Nota importante da verificare col docente

Il vincolo di copertura minima per il Caso B, cosi' come specificato nel
prompt fornito ("HARD CONSTRAINT 7"), richiede letteralmente **almeno 2
lavoratori standard E almeno 1 specializzato** per turno. Questo pero' è
in tensione con la descrizione testuale della traccia ("un turno puo'
essere coperto da un lavoratore standard e due specializzati"), che
implicherebbe che lo specializzato possa *sostituire* uno standard (quindi
basterebbero 3 persone totali, di cui almeno 1 specializzato, non
necessariamente 2 "vere" standard). Ho implementato la versione letterale
del template come richiesto ("i vincoli HARD devono rimanere IDENTICI").
Nei run osservati (vedi relazione, sezione 6) questo non ha causato
infeasibility pratica, ma vale la pena chiarirlo prima della consegna.
