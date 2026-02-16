"""
Image Library Module
Handles scanning, filtering, and managing generated images with embedded metadata.
"""

import os
import json
import time
from datetime import datetime
from typing import Optional
from PIL import Image

import modules.config
import args_manager
from modules.meta_parser import read_info_from_image, get_metadata_parser, MetadataScheme
from modules.flags import OutputFormat


class ImageLibrary:
    """Manages the image library with scanning, filtering, and tag operations."""
    
    CACHE_EXPIRY_SECONDS = 60  # Cache expires after 60 seconds
    
    def __init__(self):
        self._cache = None
        self._cache_timestamp = 0
        self._tag_index = {}
        self._tag_index_timestamp = 0
    
    def get_output_folder(self) -> str:
        """Get the output folder path."""
        try:
            return modules.config.path_outputs
        except Exception:
            return None
    
    def scan_images(self, force_refresh: bool = False) -> list[dict]:
        """
        Scan the output folder for images with metadata.
        Returns a list of image info dictionaries.
        Uses caching to avoid repeated filesystem scans.
        """
        # Check cache
        if not force_refresh and self._cache is not None:
            if time.time() - self._cache_timestamp < self.CACHE_EXPIRY_SECONDS:
                return self._cache
        
        output_folder = self.get_output_folder()
        if not output_folder or not os.path.isdir(output_folder):
            return []
        
        images = []
        supported_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        
        try:
            for root, dirs, files in os.walk(output_folder):
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_extensions:
                        filepath = os.path.join(root, filename)
                        try:
                            info = self.get_image_info(filepath)
                            if info:
                                images.append(info)
                        except Exception as e:
                            print(f"Error reading {filepath}: {e}")
                            continue
        except Exception as e:
            print(f"Error scanning output folder: {e}")
            return []
        
        # Sort by modification time, newest first
        images.sort(key=lambda x: x.get('mtime', 0), reverse=True)
        
        # Update cache
        self._cache = images
        self._cache_timestamp = time.time()
        
        return images
    
    def get_image_info(self, filepath: str) -> Optional[dict]:
        """
        Get information about a single image including metadata.
        """
        if not os.path.exists(filepath):
            return None
        
        try:
            stat = os.stat(filepath)
            mtime = stat.st_mtime
            
            # Get relative path from output folder
            output_folder = self.get_output_folder()
            rel_path = os.path.relpath(filepath, output_folder) if output_folder else filepath
            
            # Read metadata from image
            with Image.open(filepath) as img:
                parameters, metadata_scheme = read_info_from_image(img)
            
            # Parse metadata
            metadata = {}
            tags = []
            
            if parameters is not None and metadata_scheme is not None:
                try:
                    parser = get_metadata_parser(metadata_scheme)
                    metadata = parser.to_json(parameters)
                    tags = metadata.get('tags', [])
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(',') if t.strip()]
                except Exception as e:
                    print(f"Error parsing metadata for {filepath}: {e}")
            
            return {
                'path': filepath,
                'rel_path': rel_path,
                'filename': os.path.basename(filepath),
                'mtime': mtime,
                'date': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d'),
                'time': datetime.fromtimestamp(mtime).strftime('%H:%M:%S'),
                'tags': tags,
                'metadata': metadata,
                'has_metadata': bool(metadata)
            }
        except Exception as e:
            print(f"Error getting image info for {filepath}: {e}")
            return None
    
    def filter_images(
        self,
        images: list[dict],
        tags: list[str] = None,
        date: str = None,
        search: str = None,
        sort_by: str = 'date',
        sort_desc: bool = True
    ) -> list[dict]:
        """
        Filter images by tags, date, and search text.
        """
        result = images
        
        # Filter by tags
        if tags and len(tags) > 0:
            result = [
                img for img in result
                if any(tag in img.get('tags', []) for tag in tags)
            ]
        
        # Filter by date
        if date:
            result = [
                img for img in result
                if img.get('date', '').startswith(date)
            ]
        
        # Filter by search text (in prompt)
        if search:
            search_lower = search.lower()
            result = [
                img for img in result
                if search_lower in img.get('metadata', {}).get('prompt', '').lower()
                or search_lower in img.get('metadata', {}).get('negative_prompt', '').lower()
            ]
        
        # Sort
        if sort_by == 'date':
            result.sort(key=lambda x: x.get('mtime', 0), reverse=sort_desc)
        elif sort_by == 'seed':
            result.sort(
                key=lambda x: x.get('metadata', {}).get('seed', 0) or 0,
                reverse=sort_desc
            )
        
        return result
    
    def get_all_tags(self, force_refresh: bool = False) -> dict:
        """
        Get all tags with their usage counts.
        Returns dict of {tag: count}.
        """
        if not force_refresh and self._tag_index:
            if time.time() - self._tag_index_timestamp < self.CACHE_EXPIRY_SECONDS:
                return self._tag_index
        
        images = self.scan_images(force_refresh)
        tag_counts = {}
        
        for img in images:
            for tag in img.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort by count descending
        self._tag_index = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))
        self._tag_index_timestamp = time.time()
        
        return self._tag_index
    
    def update_image_tags(self, filepath: str, tags: list[str]) -> bool:
        """
        Update tags in an image's metadata.
        This requires rewriting the image with new metadata.
        """
        if not os.path.exists(filepath):
            return False
        
        try:
            with Image.open(filepath) as img:
                # Read existing metadata
                parameters, metadata_scheme = read_info_from_image(img)
                
                if parameters is None or metadata_scheme is None:
                    print(f"No metadata found in {filepath}")
                    return False
                
                parser = get_metadata_parser(metadata_scheme)
                metadata = parser.to_json(parameters)
                
                # Update tags
                metadata['tags'] = tags
                
                # Get image format
                ext = os.path.splitext(filepath)[1].lower()
                
                # Prepare new metadata string
                # We need to convert metadata back to the format expected by the parser
                new_metadata_str = parser.to_string(self._dict_to_metadata_list(metadata))
                
                # Save with new metadata
                if ext == '.png':
                    from PIL.PngImagePlugin import PngInfo
                    pnginfo = PngInfo()
                    pnginfo.add_text('parameters', new_metadata_str)
                    pnginfo.add_text('fooocus_scheme', metadata_scheme.value)
                    img.save(filepath, pnginfo=pnginfo)
                elif ext in ['.jpg', '.jpeg', '.webp']:
                    from modules.meta_parser import get_exif
                    exif = get_exif(new_metadata_str, metadata_scheme.value)
                    if ext == '.webp':
                        img.save(filepath, quality=95, lossless=False, exif=exif)
                    else:
                        img.save(filepath, quality=95, exif=exif)
                else:
                    img.save(filepath)
                
                # Invalidate cache
                self._cache = None
                self._tag_index = {}
                
                return True
        except Exception as e:
            print(f"Error updating tags for {filepath}: {e}")
            return False
    
    def _dict_to_metadata_list(self, metadata: dict) -> list:
        """Convert metadata dict to list of (label, key, value) tuples."""
        # This is a simplified version - the actual implementation would need
        # to match the format expected by the metadata parser
        result = []
        key_to_label = {
            'prompt': 'Prompt',
            'negative_prompt': 'Negative Prompt',
            'styles': 'Styles',
            'seed': 'Seed',
            'steps': 'Steps',
            'cfg_scale': 'CFG Scale',
            'sampler': 'Sampler',
            'scheduler': 'Scheduler',
            'base_model': 'Base Model',
            'refiner_model': 'Refiner Model',
            'tags': 'Tags'
        }
        
        for key, value in metadata.items():
            label = key_to_label.get(key, key)
            result.append((label, key, value))
        
        return result
    
    def load_settings_from_image(self, filepath: str) -> Optional[dict]:
        """
        Load generation settings from an image for use in the UI.
        Returns a dict with all generation parameters.
        """
        info = self.get_image_info(filepath)
        if not info or not info.get('has_metadata'):
            return None
        
        return info.get('metadata', {})
    
    def delete_image(self, filepath: str) -> bool:
        """
        Delete an image file.
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                # Invalidate cache
                self._cache = None
                self._tag_index = {}
                return True
            return False
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")
            return False
    
    def get_image_count(self) -> int:
        """Get total number of images in the library."""
        return len(self.scan_images())
    
    def clear_cache(self):
        """Clear the cache to force refresh on next scan."""
        self._cache = None
        self._tag_index = {}


# Global instance
library = ImageLibrary()


def get_library() -> ImageLibrary:
    """Get the global ImageLibrary instance."""
    return library


def scan_images(force_refresh: bool = False) -> list[dict]:
    """Convenience function to scan images."""
    return library.scan_images(force_refresh)


def get_all_tags(force_refresh: bool = False) -> dict:
    """Convenience function to get all tags."""
    return library.get_all_tags(force_refresh)


def filter_images(images: list[dict], tags: list[str] = None, date: str = None, search: str = None) -> list[dict]:
    """Convenience function to filter images."""
    return library.filter_images(images, tags, date, search)


def load_settings_from_image(filepath: str) -> Optional[dict]:
    """Convenience function to load settings from an image."""
    return library.load_settings_from_image(filepath)


def update_image_tags(filepath: str, tags: list[str]) -> bool:
    """Convenience function to update image tags."""
    return library.update_image_tags(filepath, tags)


def delete_image(filepath: str) -> bool:
    """Convenience function to delete an image."""
    return library.delete_image(filepath)