# Checkpoints

This folder is intentionally tracked without model binaries.

`MultiTaskPerceptionModel` in [models/multitask.py](../models/multitask.py) auto-downloads these files when missing:

- `classifier.pth` (Google Drive ID: `1c-3v_lRaMJiS28rK1WQOuP31qgBwtMUl`)
- `localizer.pth` (Google Drive ID: `150uYxzErRgr9KOy3Z94MwG-0YExOlJZE`)
- `unet.pth` (Google Drive ID: `1z7JK5cHYOAicmNkPpHvz6mg4R67c9Rl7`)

Keep this directory in git, but do not commit `.pth` files.
