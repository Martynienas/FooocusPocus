<div align=center>
<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/483fb86d-c9a2-4c20-997c-46dafc124f25">
</div>

# FooocusPocus

**A Quality-of-Life Enhanced Fork of [Fooocus](https://github.com/lllyasviel/Fooocus)**

[>>> Click Here to Install FooocusPocus <<<](#download)

## About This Fork

FooocusPocus is a fork of the amazing [Fooocus](https://github.com/lllyasviel/Fooocus) project by [lllyasviel](https://github.com/lllyasviel). 

**Our Goal:** Keep all existing features while enhancing quality of life and user experience. We focus on making Fooocus more comfortable to use without changing its core functionality.

**For in-depth capabilities and documentation, please visit the [original Fooocus repository](https://github.com/lllyasviel/Fooocus).**

---

## What's New in FooocusPocus

### 🎛️ Configuration Tab (New!)

A dedicated Configuration tab for managing all application settings in one place:

- **Model Folders Management**
  - Add/remove checkpoint folders without restarting
  - Add/remove LoRA folders without restarting
  - Auto-reload models when folders are added
  - Persistent folder settings across restarts

- **Default Settings**
  - Default base model, refiner model, and VAE
  - Default generation settings (steps, CFG, sharpness, sampler, scheduler)
  - Default aspect ratio and output format
  - Default styles selection

- **Path Configuration**
  - Output path, temp path
  - Embeddings, VAE, ControlNet, Upscale model paths
  - All paths editable with reset buttons

- **Per-Setting Reset**
  - Each setting has its own "Reset" button
  - "Restore All to Defaults" for complete reset
  - Auto-save on every change

### 🖼️ UI Improvements

- **Reorganized Layout**
  - Prompt inputs moved to left panel for better workflow
  - Negative prompt repositioned for easier access
  - Expanded view improvements

- **Better Logging**
  - Enhanced console output for debugging
  - Progress indicators for model operations

### 🔧 Bug Fixes

- Fixed various Gradio errors
- Improved model folder handling
- Better error messages for invalid paths
- Graceful handling of missing/invalid folders

---

## Download

### Windows

You can download FooocusPocus from the [Releases page](https://github.com/Martynienas/FooocusPocus/releases).

After downloading, uncompress and run `run.bat`.

### System Requirements

- **Minimum:** 4GB Nvidia GPU memory (VRAM) and 8GB system memory (RAM)
- **Recommended:** 6GB+ VRAM and 16GB+ RAM for optimal performance

---

## Changes from Upstream

Below is a summary of all changes compared to the original Fooocus:

| Feature | Description |
|---------|-------------|
| Configuration Tab | New tab for managing all settings with per-setting reset buttons |
| Model Folders | Add/remove model folders dynamically without restart |
| Auto-save Settings | All configuration changes saved automatically |
| UI Layout | Prompts moved to left panel, negative prompt repositioned |
| Logging | Enhanced console output and progress indicators |
| Bug Fixes | Various Gradio error fixes and improved error handling |

---

## Original Fooocus Features

FooocusPocus includes all features from the original Fooocus:

- High-quality text-to-image generation
- GPT-2 based prompt processing
- Inpainting and outpainting
- Image prompt support
- Multiple style presets
- Upscale and variation options
- FaceSwap support
- And much more...

**For complete documentation of all features, visit the [original Fooocus repository](https://github.com/lllyasviel/Fooocus).**

---

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project inherits the same license as Fooocus. See [LICENSE](LICENSE) for details.

## Credits

- Original Fooocus by [lllyasviel](https://github.com/lllyasviel)
- FooocusPocus enhancements by contributors

---

**Note:** This is an unofficial fork. For the official Fooocus project, please visit [github.com/lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus).
