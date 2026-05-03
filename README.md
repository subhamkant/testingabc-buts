# 🕉️ Mahabharata YouTube Bot

Fully automated AI-powered YouTube channel that posts **Mahabharata stories &
motivational content** in Hindi and English — **4 videos per day, 100% free.**

---

## 📦 Tech Stack (All Free)

| Step | Tool | Cost |
|---|---|---|
| 📜 Script | Gemini 1.5 Flash API | Free tier |
| 🎙️ Voice | Edge TTS (Microsoft) | Free |
| 🖼️ Images | Pollinations.ai | Free, no key |
| 🎬 Video | FFmpeg | Free |
| 📤 Upload | YouTube Data API v3 | Free |
| ⏰ Schedule | GitHub Actions | Free |

---

## 🚀 Setup Guide

### Step 1 — Clone & install locally

```bash
git clone https://github.com/YOUR_USERNAME/mahabharata-bot.git
cd mahabharata-bot
pip install -r requirements.txt
sudo apt install ffmpeg   # Linux / Mac: brew install ffmpeg
```

---

### Step 2 — Get Gemini API Key (Free)

1. Go to **https://ai.google.dev/** → Click "Get API key"
2. Create a key in **Google AI Studio** (free tier: 15 req/min)
3. Copy the key — you'll need it later

---

### Step 3 — Set up YouTube API

1. Go to **https://console.cloud.google.com/**
2. Create a new project (e.g. `mahabharata-bot`)
3. Search for **"YouTube Data API v3"** → Enable it
4. Go to **OAuth consent screen**:
   - User type: External
   - Add your Gmail address as a **Test user**
5. Go to **Credentials** → Create OAuth 2.0 Client ID
   - Application type: **Desktop App**
   - Download the JSON → rename it `client_secrets.json`
6. Place `client_secrets.json` in the project root

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
python main.py en    # English video
python main.py hi    # Hindi video
```

You'll see the full pipeline run and a video upload to your channel! 🎉

---

### Step 6 — Deploy to GitHub Actions

Add these **3 secrets** to your GitHub repo:
(`Settings → Secrets and variables → Actions → New repository secret`)

| Secret Name | Value |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key from Step 2 |
| `YOUTUBE_TOKEN_B64` | Contents of `token_base64.txt` from Step 4 |
| `YOUTUBE_CLIENT_SECRETS` | Contents of `client_secrets.json` (the whole JSON) |

Push the code to GitHub. The workflow will auto-run 4× per day! ✅

---

## ⏰ Upload Schedule (IST)

| Time (IST) | Language |
|---|---|
| 12:30 AM | 🌐 English |
|  6:30 AM | 🇮🇳 Hindi |
| 12:30 PM | 🌐 English |
|  6:30 PM | 🇮🇳 Hindi |

---

## 🎬 What Each Video Contains

- 6–8 scenes, 2–3 minutes long
- Epic AI-generated images (Pollinations.ai)
- Natural Hindi or Indian English narration (Edge TTS)
- Ken Burns zoom effect on images
- Fade transitions between scenes
- Auto-generated title, description, tags & thumbnail
- Uploaded as **Public** to YouTube automatically

---

## 🛠️ Customisation

**Change topics** → Edit `STORY_TOPICS` and `MOTIVATIONAL_THEMES` in `pipeline/script_generator.py`

**Change voices** → Edit `VOICES` dict in `pipeline/tts_generator.py`
- Browse all voices: `edge-tts --list-voices`

**Change schedule** → Edit cron expressions in `.github/workflows/schedule.yml`

**Change image style** → Edit `STYLE_SUFFIX` in `pipeline/image_generator.py`

---

## ❓ Troubleshooting

**"token.pickle expired"** — Re-run `python setup_auth.py` and update `YOUTUBE_TOKEN_B64` secret.

**"Quota exceeded" on YouTube API** — Free tier = 10,000 units/day. Each upload ≈ 1,600 units. You get ~6 uploads/day free.

**Pollinations image timeout** — The script retries 4× automatically. Slow internet or API load may cause delays.

**FFmpeg not found** — Install with `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (Mac).

---

## 📁 Project Structure

```
mahabharata-bot/
├── main.py                   # Main pipeline orchestrator
├── setup_auth.py             # One-time YouTube OAuth setup
├── requirements.txt
├── client_secrets.json       # (you create this — not committed to git)
├── token.pickle              # (auto-generated — not committed to git)
├── pipeline/
│   ├── script_generator.py   # Gemini AI script generation
│   ├── tts_generator.py      # Edge TTS voiceover
│   ├── image_generator.py    # Pollinations.ai images
│   ├── video_assembler.py    # FFmpeg video assembly
│   └── youtube_uploader.py   # YouTube Data API upload
└── .github/
    └── workflows/
        └── schedule.yml      # GitHub Actions scheduler
```
