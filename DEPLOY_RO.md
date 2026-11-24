# Cum faci proiectul live — pas cu pas

Ai două părți:
- **Aplicația web** (Field Notes, cu cont, istoric, documente, agent) — e deja live: https://fable-powered-plan-studio.lovable.app
- **Versiunea Python** (arhiva `field-notes-python_v3.zip`) — o pui pe GitHub și o faci live pe Streamlit. Durează ~15 minute.

---

## Ce îți trebuie
1. Un cont **GitHub** (github.com, gratuit).
2. O cheie **Anthropic**: console.anthropic.com → *API Keys* → *Create Key*. Pui și câțiva dolari credit la *Billing*. Cheia începe cu `sk-ant-`. **N-o arăta nimănui și n-o pune în cod.**
3. O **parolă** inventată de tine pentru pagina demo (o dai doar celor care trebuie să testeze).

---

## Pasul 1 — Pune codul pe GitHub
1. Dezarhivează `field-notes-python_v3.zip`. Vei avea dosarul `field-notes-python`.
2. Pe github.com: **+** (sus-dreapta) → **New repository**.
   - Nume: `field-notes-python`
   - **Public** (ca recrutorii să-l poată vedea)
   - Nu bifa nimic altceva → **Create repository**.
3. Pe pagina nouă apasă linkul **uploading an existing file**.
4. Deschide dosarul `field-notes-python` și **trage tot ce e înăuntru** (nu dosarul în sine) în fereastra GitHub.
   - Fișierele care încep cu punct (`.gitignore`, `.env.example`, dosarul `.streamlit`) pot fi ascunse în Windows/Mac. Activează „afișează fișierele ascunse” ca să le tragi și pe ele.
   - **Nu urca niciodată un fișier `.env`** sau `secrets.toml` cu cheia ta.
5. Jos apasă **Commit changes**. Gata — vezi README-ul afișat pe pagina repo-ului.

## Pasul 2 — Fă-l live pe Streamlit
1. Intră pe **share.streamlit.io** → **Continue with GitHub** și acceptă accesul.
2. **Create app** → **Deploy a public app from GitHub**.
3. Completează:
   - Repository: `numele-tău/field-notes-python`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: alege un nume, ex. `field-notes-filip`
4. Apasă **Advanced settings** → la **Secrets** lipește (cu valorile tale):
   ```
   APP_PASSWORD = "parola-ta-demo"
   ANTHROPIC_API_KEY = "sk-ant-..."
   MAX_INTERVIEWS_PER_VISIT = 3
   ```
5. **Deploy**. Prima pornire durează 2–3 minute (instalează pachetele).
6. Primești linkul, ex. `https://field-notes-filip.streamlit.app`.

## Pasul 3 — Verifică
1. Deschide linkul → introdu parola.
2. Fă un interviu de 3 întrebări (încearcă și **Agent-led**, și încărcarea fișierului `examples/product_brief.txt`).
3. La final trebuie să vezi sumarul, sentimentul și cuvintele-cheie.
Dacă vezi o eroare roșie cu „ANTHROPIC_API_KEY” sau „credit”, verifică cheia și creditul în consola Anthropic, apoi în Streamlit: **⋮ → Settings → Secrets**.

## Pasul 4 — Pune linkurile în CV / aplicație
- Repo GitHub: `https://github.com/numele-tău/field-notes-python`
- Demo Python: `https://field-notes-filip.streamlit.app` (+ parola, trimisă separat)
- Aplicația web: `https://fable-powered-plan-studio.lovable.app`

Opțional: pe pagina repo-ului, la **About** (rotița din dreapta), pune linkul demo-ului la *Website*.

---

## Întrebări frecvente
- **Cât costă?** GitHub și Streamlit sunt gratuite. Plătești doar apelurile la Claude: un interviu = numărul de întrebări + 1 apel (câțiva cenți).
- **Poate cineva să-mi consume creditul?** Doar cine are parola, și maximum 3 interviuri pe vizită. Schimbă parola în *Secrets* oricând.
- **Aplicația „doarme”.** Streamlit oprește aplicațiile nefolosite; primul vizitator apasă „Wake up” și așteaptă ~30 secunde.
- **Vreau să schimb ceva în cod.** Editează fișierul direct pe GitHub (creionul ✏️) → *Commit*; Streamlit se actualizează singur în ~1 minut.
- **Rulez local?** `pip install -r requirements.txt`, apoi `streamlit run streamlit_app.py` (cu cheile în `.streamlit/secrets.toml`).
