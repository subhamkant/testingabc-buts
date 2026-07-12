# source_chapters/

Raw Martial Peak web-novel chapter text — the SOURCE for faithful recap scripts.

## How to use
Drop plain-text files here, one per chapter, named **`chapter_NNN.txt`** (zero-padded):

```
source_chapters/
  chapter_001.txt
  chapter_002.txt
  chapter_003.txt
  ...
```

Each file = the full raw text of that chapter (copy-paste from your source).

## What happens
`pipeline/wuxia_script.py` reads the next `WUXIA_CHAPTERS_PER_EP` chapters
(default 3), feeds them to Gemini in segments, and writes a ~110-scene Hindi
recap to `pro_drafts/wuxia/<slug>/longform_hi.json`. It tracks progress in
`assets/wuxia_chapter_progress.json` (`next_chapter`, `episode`) and auto-advances
each run, so daily runs consume the next chapters automatically.

Generate + render in one command:
```
python wuxia_main.py --from-chapters
```
Or generate the script only (to review first):
```
python -m pipeline.wuxia_script            # writes the JSON, advances pointer
python -m pipeline.wuxia_script --no-advance
```

## Note
These `.txt` files are the copyrighted novel text — **do not commit them to a
public repo.** They're gitignored. For GHA automation, fetch them from private
storage (e.g. Cloudflare R2) at runtime.
