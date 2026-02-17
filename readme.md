# FooocusPocus

**A Quality-of-Life Enhanced Fork of [Fooocus](https://github.com/lllyasviel/Fooocus)**

[>>> Click Here to Install FooocusPocus <<<](#download)

## About This Fork

FooocusPocus is a fork of [Fooocus](https://github.com/lllyasviel/Fooocus) by [lllyasviel](https://github.com/lllyasviel).

Goal: keep core Fooocus generation behavior while improving usability, workflow, and configuration quality of life.

For full upstream capability docs, see the [original Fooocus repository](https://github.com/lllyasviel/Fooocus).

---

## Highlights

### Image Library (Major Feature)

The Image Library is an in-app browser for generated images with metadata-aware management.

- Browse generated images in a dedicated modal gallery
- Search images by prompt text
- Filter by tags
- Single-image preview with full metadata panel
- Multi-select with checkboxes for bulk operations
- Unified delete action:
  - Deletes one image when single-selected
  - Deletes multiple images when checkbox selection is active
- Edit tags for one or many images
- Optional auto-reload when opening the library

### Configuration Tab

A dedicated tab for managing runtime settings without manual config editing.

- Add/remove checkpoint folders and LoRA folders
- Auto-reload models after folder changes
- Edit key paths (output/temp/embeddings/VAE/ControlNet/upscale)
- Set generation defaults (steps/CFG/sampler/scheduler/model/style/aspect ratio/output format)
- Per-setting reset buttons and full restore-to-defaults
- Auto-save configuration changes

### UI and Workflow Improvements

- Prompts are organized for faster day-to-day use
- Better operation feedback and console logging
- Random LoRA option for exploration

---

## Prompt Syntax Reference

FooocusPocus supports both wildcard placeholders and dynamic prompt groups.

### 1) Wildcard files (`__name__`)

Use wildcard placeholders in prompts:

- `a portrait of __artist__`
- `__color__ sports car in __city__`

Wildcard files are loaded from your configured wildcards folder (`path_wildcards`), with one option per line.

### 2) Dynamic choices (`{...}`)

Use inline dynamic groups:

- Single choice: `{red|green|blue}`
- Single choice (spaces are fine): `{red | green | blue}`
- Multi-select count: `{2$$red|green|blue|yellow}`
- Multi-select range: `{1-3$$red|green|blue|yellow}`

Both positive and negative prompts support this syntax.

### 3) Read wildcards in order

When enabled, wildcard file entries are consumed deterministically by index (useful for reproducible batches).  
When disabled, wildcard entries are chosen randomly.

---

## Download

### Windows

You can download FooocusPocus from the [Releases page](https://github.com/Martynienas/FooocusPocus/releases).

After downloading, extract and run `run.bat`.

### System Requirements

- **Minimum:** 4GB Nvidia GPU VRAM and 8GB system RAM
- **Recommended:** 6GB+ VRAM and 16GB+ RAM

---

## Changes from Upstream (Summary)

| Feature | Description |
|---------|-------------|
| Image Library | In-app generated image browser with metadata, tag filter/search, multiselect, bulk delete, and tag editing |
| Configuration Tab | UI-based settings management with auto-save and reset controls |
| Dynamic Model Folders | Add/remove checkpoint and LoRA folders without restart |
| Prompt Utilities | Wildcard placeholders and dynamic prompt groups |
| Random LoRA | Optional random LoRA selection for experiments |
| UX/Logging | Improved feedback and operational visibility |

---

## Original Fooocus Features

FooocusPocus includes upstream Fooocus capabilities, including:

- High-quality text-to-image generation
- Inpainting and outpainting
- Image prompt workflows
- Style systems and model switching
- Upscale and variation workflows

For full documentation, see [Fooocus upstream](https://github.com/lllyasviel/Fooocus).

---

## Contributing

Contributions are welcome via pull requests and issues.

## License

This project inherits Fooocus licensing. See [LICENSE](LICENSE).

## Credits

- Original Fooocus by [lllyasviel](https://github.com/lllyasviel)
- FooocusPocus enhancements by contributors

---

**Note:** This is an unofficial fork. For the official project, visit [github.com/lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus).
