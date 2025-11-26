# Political Compass Party Matcher

Interactive [Streamlit](https://streamlit.io/) tool that places you on a two-axis political
compass (economic left/right and social libertarian/authoritarian) and suggests the party family
closest to your answers.

## Features

- Policy prompts stay in sync with the party program you are reading
- Live compass updates as you change each answer
- Normalized -10..10 axis scores with quadrant classification
- Rank-ordered party families plus a Plotly compass with quadrant arrows
- Detailed breakdown showing how each answer nudged your score
- Topic-based party comparison (data sourced from partiebi.com)
- Lightweight chatbot that answers questions from the loaded party programs

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate          # On Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501) and complete the survey.

### Updating the policy database

Policy prompts, party comparisons, and the chatbot all read from `data/policy_topics.json`.
Each entry contains:

- `slug`, `theme`, `prompt`, `axes`, and `question` (used for scoring/UI)
- `source` (URL reference)
- `parties`: list of `{ "name": ..., "position": ... }`

Add or edit entries in that file to incorporate new topics from partiebi.com or other sources.

## Deploying to Streamlit Cloud / GitHub

1. Initialize a repo and push it to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial political compass app"
   git branch -M main
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Create a new Streamlit deployment, point it at your repo, and set the entry point to `app.py`.
4. Add the environment variable `PYTHONPATH=.` if needed, and be sure `requirements.txt` contains `streamlit`, `plotly`, and `orjson`.
5. Click “Deploy”. Streamlit Cloud will install dependencies, run `app.py`, and provide a public URL you can share.

Whenever you update topics or code, push to GitHub and Streamlit Cloud will automatically redeploy the latest commit.

## Notes

- This compass is a heuristic for exploration, not a scientific instrument.
- Party families are generalized archetypes positioned on the same coordinate space. Consider them
  as guidance for further research rather than exact matches.

