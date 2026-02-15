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
    """Check if a style is a user-defined style."""
    user_styles = load_user_styles()
    for style in user_styles:
        if normalize_style_name(style.get('name', '')) == style_name:
            return True
    return False


def is_system_style(style_name):
    """Check if a style is a system style (cannot be deleted)."""
    # System styles are those from the default style files
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


def create_style(name, prompt='', negative_prompt=''):
    """Create a new user-defined style."""
    if not name or not name.strip():
        return False, "Style name cannot be empty!"
    
    name = name.strip()
    normalized_name = normalize_style_name(name)
    
    # Check if style already exists
    if normalized_name in sdxl_styles.styles:
        return False, f"Style '{normalized_name}' already exists!"
    
    # Load existing user styles
    user_styles = load_user_styles()
    
    # Add new style
    new_style = {
        'name': name,
        'prompt': prompt,
        'negative_prompt': negative_prompt
    }
    user_styles.append(new_style)
    
    # Save user styles
    success, message = save_user_styles(user_styles)
    if success:
        # Reload styles
        reload_styles()
        return True, f"Style '{normalized_name}' created successfully!"
    return False, message


def update_style(old_name, new_name, prompt='', negative_prompt=''):
    """Update an existing user-defined style."""
    if not new_name or not new_name.strip():
        return False, "Style name cannot be empty!"
    
    old_name = old_name.strip()
    new_name = new_name.strip()
    normalized_old = normalize_style_name(old_name)
    normalized_new = normalize_style_name(new_name)
    
    # Check if old style exists
    if normalized_old not in sdxl_styles.styles:
        return False, f"Style '{normalized_old}' does not exist!"
    
    # Check if trying to update a system style
    if is_system_style(normalized_old):
        return False, f"Cannot modify system style '{normalized_old}'!"
    
    # Load user styles
    user_styles = load_user_styles()
    
    # Find and update the style
    found = False
    for i, style in enumerate(user_styles):
        if normalize_style_name(style.get('name', '')) == normalized_old:
            user_styles[i] = {
                'name': new_name,
                'prompt': prompt,
                'negative_prompt': negative_prompt
            }
            found = True
            break
    
    if not found:
        # Style might exist but not in user styles file, add it
        user_styles.append({
            'name': new_name,
            'prompt': prompt,
            'negative_prompt': negative_prompt
        })
    
    # If renaming, check if new name already exists
    if normalized_old != normalized_new and normalized_new in sdxl_styles.styles:
        return False, f"Style '{normalized_new}' already exists!"
    
    # Save user styles
    success, message = save_user_styles(user_styles)
    if success:
        # Reload styles
        reload_styles()
        return True, f"Style '{normalized_old}' updated to '{normalized_new}' successfully!"
    return False, message


def delete_style(style_name):
    """Delete a user-defined style."""
    if not style_name:
        return False, "No style selected!"
    
    normalized_name = normalize_style_name(style_name)
    
    # Check if style exists
    if normalized_name not in sdxl_styles.styles:
        return False, f"Style '{normalized_name}' does not exist!"
    
    # Check if trying to delete a system style
    if is_system_style(normalized_name):
        return False, f"Cannot delete system style '{normalized_name}'!"
    
    # Check if it's Fooocus V2 or Random Style
    if normalized_name in ['Fooocus V2', 'Random Style']:
        return False, f"Cannot delete '{normalized_name}'!"
    
    # Load user styles
    user_styles = load_user_styles()
    
    # Remove the style
    user_styles = [s for s in user_styles if normalize_style_name(s.get('name', '')) != normalized_name]
    
    # Save user styles
    success, message = save_user_styles(user_styles)
    if success:
        # Reload styles
        reload_styles()
        return True, f"Style '{normalized_name}' deleted successfully!"
    return False, message


def reload_styles():
    """Reload all styles from files."""
    # Clear existing styles
    sdxl_styles.styles.clear()
    
    # Reload from all style files
    styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))
    styles_files = get_files_from_folder(styles_path, ['.json'])
    
    # Order: load user styles last so they can override
    for x in ['sdxl_styles_fooocus.json',
              'sdxl_styles_sai.json',
              'sdxl_styles_mre.json',
              'sdxl_styles_twri.json',
              'sdxl_styles_diva.json',
              'sdxl_styles_marc_k3nt3l.json']:
        if x in styles_files:
            styles_files.remove(x)
            styles_files.append(x)
    
    # Move user styles to the end
    if 'sdxl_styles_user.json' in styles_files:
        styles_files.remove('sdxl_styles_user.json')
        styles_files.append('sdxl_styles_user.json')
    
    for styles_file in styles_files:
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
