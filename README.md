# Martini Sports Lab

Live site: **https://lmartini27.github.io**

Weekly NFL fantasy projections, graded in public against a simple benchmark,
plus four longer research projects. Research code lives in
[sports-anaylsis-portfolio](https://github.com/lmartini27/sports-anaylsis-portfolio).

## How it works

- `pipeline/update.py` downloads the latest [nflverse](https://github.com/nflverse) stats,
  grades last week's projections, and writes next week's to `data/`.
- `.github/workflows/weekly-update.yml` runs it every Wednesday and commits the results.
  The commit history is the public record of when each week's projections were posted.
- `index.html` is the whole site. It reads the JSON files in `data/`.

## Run it locally

```
pip install -r requirements.txt
python pipeline/update.py --season 2026
python -m http.server 8000      # then open http://localhost:8000
```
