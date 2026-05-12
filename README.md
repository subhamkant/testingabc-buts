# 🕉️ testing fun bot

test posts **Mahabharata stories &
motivational content** in Hindi and English — ****

---

## 📦 Tech Stack (All Free)

| Step | Tool | Cost |
|---|---|---|
| 📜 Script | Gemini 1.5 Flash API | Free tier |
| 🎙️ Voice | Edge TTS (Microsoft) | Free |

---

## 🚀 Setup Guide

### Step 1 — Clone & install locally

```bash

```

---

### Step 2 — Get Gemini API Key (Free)

test
---

### Step 3 — Set up YouTube API

test3

---

### Step 4 — Authenticate with YouTube (One-time)

```bash
python setup_auth.py
```

This opens a browser window. Log in with the YouTube channel account.
After login:
- `token.pickle` is created locally
- `token_base64.txt` is created with the base64 version

---

### Step 5 — Test locally

```bash
test
```

---

### Step 6 — Deploy to GitHub Actions

Add these **3 secrets** to your GitHub repo:
(`Settings → Secrets and variables → Actions → New repository secret`)

| Secret Name | Value |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key from Step 2 |

---

## 🎬 What Each Video Contains

college project

---

## 🛠️ Customisation

---

## ❓ Troubleshooting

---

## 📁 Project Structure

```
mahabharata-bot/
├── main.py                   # Main pipeline orchestrator
├── setup_auth.py             # test setup
├── requirements.txt
├── client_secrets.json       # (you create this — not committed to git)
├── token.pickle              # (auto-generated — not committed to git)
├── pipeline/
│   ├── script_generator.py   # Gemini script generation
│   ├── tts_generator.py      # Edge TTS voiceover
│  # FFmpeg video assembly
│   └── test.py   # test Data API upload
└── .github/
    └── workflows/
        └── schedule.yml      # GitHub Actions scheduler
```
