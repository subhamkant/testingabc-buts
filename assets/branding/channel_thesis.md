# Vyasa AI — Channel Thesis

> **"Every hero in Mahabharata destroyed someone."**

Chosen 2026-05-23 as the recurring worldview anchor for the Mahabharata channel. This single sentence is the lens through which every video frames its moral.

## Where this thesis lives in the pipeline

| Location | What it does |
|---|---|
| [pipeline/script_generator.py](../../pipeline/script_generator.py) system-prompt opening (line ~2542) | Every script written by the LLM is told to frame the story through this lens — what the hero destroyed, lost, or broke |
| [pipeline/youtube_uploader.py](../../pipeline/youtube_uploader.py) `_PINNED_CHANNEL_THESIS` | The thesis appears as a tagline in every per-video pinned comment, between the subscribe line and the hashtags |
| YouTube channel About section | Manually maintained by the channel owner — the public-facing positioning |

## The thinking behind it

Three alternative theses were considered and rejected:

- *"Mahabharata is not history — it's psychology."* (universal but less polarizing)
- *"Every victory in Mahabharata cost something terrible."* (most coherent with the existing cost-over-pain rubric)
- *"Dharma isn't right vs wrong — it's wrong vs less wrong."* (most controversial)

The chosen thesis won because it's the **most accusatory** of the four — every hero is implicated. It pairs naturally with the Phase 26 Emotional Violence Layer (named accusations, named identity wounds) and the Phase 21-Lite-Plus tribal-split quotable_line (often inverting a hero's morality).

## What this means for content selection

Future videos should center on what the hero broke, not what they achieved:

- **Bhishma's vow** destroyed the Kuru lineage's future, not "saved the kingdom"
- **Arjuna's victory** destroyed Karna and a piece of Arjuna's own conscience
- **Krishna's strategy** destroyed warriors' codes of honor in service of the cosmic win
- **Yudhishthira's truth** destroyed Drona — the most "dharmic" character broke the most reluctantly
- **Karna's loyalty** destroyed him AND his birth-mother's peace
- **Draupadi's anger** destroyed Hastinapur, even as she was the one most wronged

No video presents a hero as purely admirable. The **cost** is the point.

## What this means for thumbnail/title direction

- Titles should NAME what was destroyed: `भीष्म ने हस्तिनापुर बचाया नहीं… खत्म किया` rather than `भीष्म की प्रतिज्ञा`
- Thumbnails should show the FACE OF THE COST — the pain on the perpetrator's face, the wound on the victim's. Not heroic stoicism.
- Pinned-comment quotable lines should INVERT the popular sympathy or ACCUSE the hero by name.

## What this is NOT

- Not "Mahabharata bashing." The destruction is real, named in the epic itself, and the show of consequence is what makes the epic literature rather than legend.
- Not contrarian for its own sake. We don't invent destruction — we refuse to look away from destruction that's already in the source text.
- Not anti-dharma. The thesis presupposes that dharma is hard *because* every dharma-choice costs someone.

## When to revisit this

Locked at 2026-05-23. Revisit only when:
- Channel has 5K+ subscribers and the thesis has demonstrably shaped audience expectations
- An alternative thesis becomes clearly more powerful from analytics signal
- The user explicitly wants to evolve the positioning
