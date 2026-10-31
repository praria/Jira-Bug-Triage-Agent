# Jira Defect Triage Agent

A local agent that helps a reviewer triage a new Jira bug by comparing it with bugs already stored from the same project. It embeds each bug’s summary and description, finds the closest stored defects, and writes a recommendation with reasoning. The recommendation is posted to Jira only after the reviewer approves it.

The dashboard is a Streamlit app. Embeddings are computed locally. The vector database is a Chroma directory on disk.

## What happens when you triage a bug

1. You enter a Jira issue key, such as `FDE-63`, on the Triage page.
2. The app loads that Bug from Jira Cloud and splits the description into the narrative, the Logs section, and the Root Cause section.
3. It embeds the summary and the narrative description.
4. It searches the local Chroma collection and returns the three closest stored bugs. The bug you entered is left out of that list.
5. It writes a recommendation:
   - **Clear match.** The closest stored bug is at least 60% similar. Gemini writes the recommendation from that bug’s stored root cause and tells you to confirm it, then apply it here.
   - **No clear match.** Every stored bug is below 60%. The recommendation tells you not to reuse a historical root cause and gives investigation steps.
6. You read the recommendation. **Approve and post to Jira** adds it as a comment on that ticket. Nothing is written to Jira before that click.

## Why the root cause is not embedded

Retrieval answers “which past bugs had the same symptoms?” The summary and the narrative description are the symptoms. The root cause is the answer returned after a match, so it is stored beside the vector as metadata and shown in the recommendation.

A new ticket often has no confirmed root cause yet. Leaving that text out of the vector keeps the comparison on the wording of the problem. The Logs section is removed from the embedded text in the same way.

## Dashboard

Start it from the project root:

```bash
streamlit run dashboard.py
```

Open http://localhost:8501.

| Page | What it shows |
| --- | --- |
| Triage | A defect id field, the loaded bug, the three closest stored defects, the recommendation, and the approve button. |
| Embeddings | A map of the stored bugs, the index table, and a bar chart of the five bugs closest to the one you select. Similarity runs from 0% to 100%. |
| Chroma database | Health of the local database: collection, record count, dimensions, distance, metadata, and the stored document for each bug. |

The embedding map compresses each 384-number vector onto two axes so the collection can be drawn. Nearby points tend to have similar wording. The percentage on the Triage page and the Closest bugs chart is the similarity used for recommendations.

The raw 384 numbers are behind **The 384 numbers behind this comparison** on the Embeddings page. They do not need to be read one by one.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`. Do not commit that file.

| Setting | Required | Purpose |
| --- | --- | --- |
| `JIRA_BASE_URL` | yes | Jira Cloud site, for example `https://your-company.atlassian.net` |
| `JIRA_EMAIL` | yes | Atlassian account email |
| `JIRA_API_TOKEN` | yes | API token. Create one with `read:jira-work`, `read:jira-user`, and `write:jira-work`. |
| `JIRA_PROJECT_KEY` | yes | Project to pull, for example `FDE` |
| `JIRA_ISSUE_TYPE` | no | Defaults to `Bug` |
| `JIRA_ROOT_CAUSE_LABEL` | no | Description heading for the cause. Defaults to `Root Cause`. `JIRA_ROOT_CAUSE_FIELD` is accepted as the same setting. |
| `EMBEDDING_MODEL` | no | Defaults to `all-MiniLM-L6-v2` |
| `CHROMA_PATH` | no | Defaults to `.chroma` |
| `GEMINI_API_KEY` | yes | API key from [Google AI Studio](https://aistudio.google.com/apikey). Gemini phrases a clear match. |
| `GEMINI_MODEL` | yes | Gemini model name. Set `gemini-3.5-flash`. |

Jira descriptions are Atlassian document format. A heading or a line such as `Root Cause:` or `Logs:` is pulled out of the narrative. The dump file stores those sections as labels:

```json
{
  "summary": "Inventory oversell during flash sale",
  "description": "Multiple customers successfully purchased the same SKU despite stock reaching zero.",
  "project_key": "FDE",
  "priority": "Highest",
  "labels": [
    "logs: ERROR [InventoryService] - OptimisticLockException on SKU_9932: version mismatch",
    "root_cause: Database transaction isolation level set to READ COMMITTED instead of SERIALIZABLE."
  ]
}
```

`priority` is Jira’s own name, such as Highest, High, Medium, or Low.

## Build the local index

From the project root, with `.env` filled in:

```bash
python pull_defects.py
python index_defects.py
```

`pull_defects.py` writes `defectdump.json`. `index_defects.py` pulls the same bugs, embeds the summary and description, and upserts them into `.chroma`. Re-running the indexer replaces a bug that is already stored under the same issue key.

The first index downloads `all-MiniLM-L6-v2`. After that, the model is cached on the machine.

## Use Gemini for the recommendation

Gemini writes the recommendation when the closest stored bug is at least 60% similar. It receives the new bug and the three retrieved bugs, and it must stay on the root cause that was retrieved. It does not search Jira or Chroma.

1. Open [Google AI Studio](https://aistudio.google.com/apikey) and create an API key.
2. Put the key and model in `.env`:

```bash
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash
```

3. Install the app dependencies. `requirements.txt` includes `google-genai`, which is the client that calls Gemini:

```bash
pip install -r requirements.txt
```

4. Start the dashboard, or restart it if it is already running. The Triage page reads `.env` when Streamlit starts, so a key added later is used only after you stop the app and run `streamlit run dashboard.py` again.
5. On the Triage page, enter a bug key and click **Triage**. When the closest stored bug is at least 60% similar, Gemini writes the headline, reasoning, and steps. Read them, then click **Approve and post to Jira**. That posts the text on the page as a comment.

A match below 60% does not call Gemini. The page shows investigation steps and tells you not to copy a historical root cause.

If Gemini returns an empty or unusable reply, the page keeps the stored root cause and shows a notice that names the failure.

## Layout

```
dashboard.py                 Streamlit entry point
app_pages/                   Triage, Embeddings, and Chroma database pages
pull_defects.py              Write defectdump.json
index_defects.py             Fill the local Chroma collection
src/jira_triage/jira/        Jira client, description parsing, and issue lookup
src/jira_triage/rag/embeddings.py    Summary and description to a vector
src/jira_triage/rag/retrieval.py     Closest stored bugs for one new bug
src/jira_triage/rag/augmentation.py  Recommendation, reasoning, and Gemini prompt
src/jira_triage/store/       Chroma collection named defects
src/jira_triage/indexing.py  Pull, embed, and upsert
tests/                       Unit tests
```

Each Chroma record uses the Jira key as its id, stores `defect_id` and `root_cause` as metadata, and stores the embedded summary and description as the document. Vectors are 384 dimensions and compared with cosine distance.

## Tests

```bash
.venv/bin/pytest
```
