# Discogs-VI Embeddings Dataset

A music information retrieval (MIR) dataset derived from the
[MTG Discogs-VI](https://github.com/MTG/discogs-vi-yt) dataset,
extended with audio embeddings, metadata, and album art.
Built as part of ongoing research into musical "vibes" and cosine similarity.

---

## Dataset Components

| Component | Location | Access |
|---|---|---|
| Audio files (zipped, ~XXX GB) | [Nextcloud](YOUR_NEXTCLOUD_SHARE_LINK) | Password-protected — for maintainers' research purposes only |
| Metadata (JSONL/Parquet) | [Hugging Face](https://huggingface.co/datasets/YOUR_ORG/discogs-vi-embeddings) | Public |
| Album art (JPEG) | [Hugging Face](https://huggingface.co/datasets/YOUR_ORG/discogs-vi-embeddings) | Public |

---

## Data Collection

This repository currently hosts a dataset derived from the
**2024 release** of Discogs-VI. A 2026 recreation built from the
most recent Discogs metadata dump is in active development —
see our [fork](YOUR_MTG_FORK_LINK) for collection scripts and updates.
The original dataset and methodology are described in:

> Correya et al., *Discogs-VI: A musical version identification
> dataset based on Discogs metadata*, ISMIR 2024.

Audio files were sourced from YouTube via the official Discogs-VI
YouTube IDs. **Audio files are not redistributed publicly** in
compliance with copyright law and in keeping with the MTG dataset's
own distribution policy.

---

## Loading the Metadata (Python)

```python
from datasets import load_dataset

ds = load_dataset("YOUR_ORG/discogs-vi-embeddings")
```

---

## License

Metadata and album art are released under
CC BY-NC 4.0.
Audio files are not redistributed — see above.