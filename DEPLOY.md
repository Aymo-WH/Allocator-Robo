# Deploying the Goal-Allocator app

This app shows your deposits and holdings, so treat it like online banking.

## Run locally first (recommended before hosting)

```bash
pip install -r requirements.txt
python src/data.py                 # build price history (once; refresh periodically)
streamlit run app.py               # opens in your browser at localhost:8501
```

Your ledger is saved to `data/ledger.json` on your machine — gitignored, never
committed.

## Hosting (Streamlit Community Cloud — free)

Before you deploy anything with your financial data, three non-negotiables:

1. **Make the GitHub repo PRIVATE.** Settings → General → Danger Zone → Change
   visibility → Private. (Code can be public; your *data* must never be.)
2. **Set a password.** In the Streamlit Cloud app settings → Secrets, add:
   ```toml
   APP_PASSWORD = "choose-a-strong-password"
   ```
   The app stays locked until this is entered. With no secret set, it runs open
   (fine for localhost only).
3. **Understand the persistence limit.** Streamlit Cloud has an *ephemeral*
   filesystem — `data/ledger.json` will reset when the app restarts. For hosted
   use your ledger must live in durable storage. Options, simplest first:
   - **Download/upload your `ledger.json`** each session (a button can be added).
   - **A free hosted DB** (e.g. Supabase / a Google Sheet via API) wired into
     `account.load_ledger`/`save_ledger`. Ask and I'll add a backend adapter.

### Steps
1. Push this repo (private) to GitHub.
2. Go to share.streamlit.io → New app → pick the repo → main file `app.py`.
3. Add the `APP_PASSWORD` secret.
4. Deploy.

> Until durable storage is wired in (point 3), prefer running locally — your
> ledger persists reliably there. Hosting is great for *viewing*; for *recording*
> deposits/fills, local is currently safer.
