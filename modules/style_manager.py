import os
import json
import re
import gradio as gr
from modules.extra_utils import get_files_from_folder
from modules import sdxl_styles


# Path to user-defined styles
user_styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/sdxl_styles_user.json'))


def get_all_styles():
    """Get all available styles including system and user styles."""
    return sdxl_styles.legal_style_names


def get_style_details(style_name):
    """Get the prompt and negative prompt for a specific style."""
    if style_name in sdxl_styles.styles:
        prompt, negative_prompt = sdxl_styles.styles[style_name]
        return prompt, negative_prompt
    return '', ''


def load_user_styles():
    """Load user-defined styles from the user styles file."""
    if os.path.exists(user_styles_path):
        try:
            with open(user_styles_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f'Failed to load user styles: {e}')
    return []


def save_user_styles(styles):
    """Save user-defined styles to the user styles file."""
    try:
        os.makedirs(os.path.dirname(user_styles_path), exist_ok=True)
        with open(user_styles_path, 'w', encoding='utf-8') as f:
            json.dump(styles, f, indent=4, ensure_ascii=False)
        return True, "Styles saved successfully!"
    except Exception as e:
        return False, f"Failed to save styles: {e}"


def normalize_style_name(name):
    """Normalize a style name to match the format used in sdxl_styles."""
    name = name.replace('-', ' ')
    words = name.split(' ')
    words = [w[:1].upper() + w[1:].lower() for w in words]
    name = ' '.join(words)
    name = name.replace('3d', '3D')
    name = name.replace('Sai', 'SAI')
    name = name.replace('Mre', 'MRE')
    return name


def is_user_style(style_name):
    """Check if a style is a user-defined style (exists in user styles file)."""
    user_styles = load_user_styles()
    for style in user_styles:
        if normalize_style_name(style.get('name', '')) == style_name:
            return True
    return False


def is_system_style(style_name):
    """Check if a style originates from a system style file."""
    system_files = [
        'sdxl_styles_fooocus.json',
        'sdxl_styles_sai.json',
        'sdxl_styles_mre.json',
        'sdxl_styles_twri.json',
        'sdxl_styles_diva.json',
        'sdxl_styles_marc_k3nt3l.json'
    ]
    
    styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))
    
    for styles_file in system_files:
        filepath = os.path.join(styles_path, styles_file)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for entry in json.load(f):
                        if normalize_style_name(entry.get('name', '')) == style_name:
                            return True
            except:
                pass
    return False


def is_overridden_system_style(style_name):
    """Check if a system style is overridden by a user style with the same name."""
    return is_system_style(style_name) and is_user_style(style_name)


def save_style(name, prompt='', negative_prompt=''):
    """Save a style (create new or update existing). 
    User styles with same name as system styles will override/hide them."""
    if not name or not name.strip():
        return False, "Style name cannot be empty!"
    
    name = name.strip()
    normalized_name = normalize_style_name(name)
    
    # Load existing user styles
    user_styles = load_user_styles()
    
    # Check if this is an update to an existing user style
    existing_index = None
    for i, style in enumerate(user_styles):
        if normalize_style_name(style.get('name', '')) == normalized_name:
            existing_index = i
            break
    
    if existing_index is not None:
        # Update existing user style
        user_styles[existing_index] = {
            'name': name,
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }
    else:
        # Add new style (may override system style with same name)
        user_styles.append({
            'name': name,
            'prompt': prompt,
            'negative_prompt': negative_prompt
        })
    
    # Save user styles
    success, message = save_user_styles(user_styles)
    if success:
        reload_styles()
        if existing_index is not None:
            return True, f"Style '{normalized_name}' updated successfully!"
        else:
            return True, f"Style '{normalized_name}' created successfully!"
    return False, message


def delete_style(style_name):
    """Delete a user-defined style. This will reveal the system style if it was overridden."""
    if not style_name:
        return False, "No style selected!"
    
    normalized_name = normalize_style_name(style_name)
    
    # Check if it's Fooocus V2 or Random Style
    if normalized_name in ['Fooocus V2', 'Random Style']:
        return False, f"Cannot delete '{normalized_name}'!"
    
    # Load user styles
    user_styles = load_user_styles()
    
    # Check if this is a user style
    is_user = False
    for style in user_styles:
        if normalize_style_name(style.get('name', '')) == normalized_name:
            is_user = True
            break
    
    if not is_user:
        return False, f"Style '{normalized_name}' is not a user style and cannot be deleted!"
    
    # Remove the style from user styles
    user_styles = [s for s in user_styles if normalize_style_name(s.get('name', '')) != normalized_name]
    
    # Save user styles
    success, message = save_user_styles(user_styles)
    if success:
        reload_styles()
        # Check if a system style will be revealed
        if is_system_style(normalized_name):
            return True, f"User style '{normalized_name}' deleted. System style is now visible!"
        return True, f"Style '{normalized_name}' deleted successfully!"
    return False, message


def reload_styles():
    """Reload all styles from files. User styles override system styles with same name."""
    # Clear existing styles
    sdxl_styles.styles.clear()
    
    # Reload from all style files
    styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))
    styles_files = get_files_from_folder(styles_path, ['.json'])
    
    # Order: load system styles first, then user styles last so they override
    system_files = [
        'sdxl_styles_fooocus.json',
        'sdxl_styles_sai.json',
        'sdxl_styles_mre.json',
        'sdxl_styles_twri.json',
        'sdxl_styles_diva.json',
        'sdxl_styles_marc_k3nt3l.json'
    ]
    
    # Sort: system files first, then others, user file last
    ordered_files = []
    for sf in system_files:
        if sf in styles_files:
            ordered_files.append(sf)
    
    for sf in styles_files:
        if sf not in ordered_files and sf != 'sdxl_styles_user.json':
            ordered_files.append(sf)
    
    # User file last to override system styles
    if 'sdxl_styles_user.json' in styles_files:
        ordered_files.append('sdxl_styles_user.json')
    
    for styles_file in ordered_files:
        try:
            with open(os.path.join(styles_path, styles_file), encoding='utf-8') as f:
                for entry in json.load(f):
                    name = sdxl_styles.normalize_key(entry['name'])
                    prompt = entry['prompt'] if 'prompt' in entry else ''
                    negative_prompt = entry['negative_prompt'] if 'negative_prompt' in entry else ''
                    sdxl_styles.styles[name] = (prompt, negative_prompt)
        except Exception as e:
            print(str(e))
            print(f'Failed to load style file {styles_file}')
    
    # Update legal style names
    sdxl_styles.style_keys = list(sdxl_styles.styles.keys())
    sdxl_styles.legal_style_names = [sdxl_styles.fooocus_expansion, sdxl_styles.random_style_name] + sdxl_styles.style_keys
    
    # Update style sorter
    import modules.style_sorter as style_sorter
    style_sorter.all_styles = sdxl_styles.legal_style_names


def get_styles_for_dropdown():
    """Get list of styles for dropdown selection."""
    return sdxl_styles.legal_style_names


def get_user_styles_for_dropdown():
    """Get list of user-defined styles for dropdown selection."""
    user_styles = load_user_styles()
    return [normalize_style_name(s.get('name', '')) for s in user_styles if s.get('name')]
