// based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/script.js
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 */
function get_uiCurrentTab() {
    return gradioApp().querySelector('#tabs > .tab-nav > button.selected');
}

/**
 * Get the first currently visible top-level UI tab content (e.g. the div hosting the "txt2img" UI).
 */
function get_uiCurrentTabContent() {
    return gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
}

var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 * @param callback
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedOnLoaded && gradioApp().querySelector('#generate_button')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
    initStylePreviewOverlay();
});

var onAppend = function(elem, f) {
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            if (m.addedNodes.length) {
                f(m.addedNodes);
            }
        });
    });
    observer.observe(elem, {childList: true});
}

function addObserverIfDesiredNodeAvailable(querySelector, callback) {
    var elem = document.querySelector(querySelector);
    if (!elem) {
        window.setTimeout(() => addObserverIfDesiredNodeAvailable(querySelector, callback), 1000);
        return;
    }

    onAppend(elem, callback);
}

/**
 * Show reset button on toast "Connection errored out."
 */
addObserverIfDesiredNodeAvailable(".toast-wrap", function(added) {
    added.forEach(function(element) {
         if (element.innerText.includes("Connection errored out.")) {
             window.setTimeout(function() {
                document.getElementById("reset_button").classList.remove("hidden");
                document.getElementById("generate_button").classList.add("hidden");
                document.getElementById("skip_button").classList.add("hidden");
                document.getElementById("stop_button").classList.add("hidden");
            });
         }
    });
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', function(e) {
    const isModifierKey = (e.metaKey || e.ctrlKey || e.altKey);
    const isEnterKey = (e.key == "Enter" || e.keyCode == 13);

    if(isModifierKey && isEnterKey) {
        const generateButton = gradioApp().querySelector('button:not(.hidden)[id=generate_button]');
        if (generateButton) {
            generateButton.click();
            e.preventDefault();
            return;
        }

        const stopButton = gradioApp().querySelector('button:not(.hidden)[id=stop_button]')
        if(stopButton) {
            stopButton.click();
            e.preventDefault();
            return;
        }
    }
});

function initStylePreviewOverlay() {
    let overlayVisible = false;
    const samplesPath = document.querySelector("meta[name='samples-path']").getAttribute("content")
    const overlay = document.createElement('div');
    const tooltip = document.createElement('div');
    tooltip.className = 'preview-tooltip';
    overlay.appendChild(tooltip);
    overlay.id = 'stylePreviewOverlay';
    document.body.appendChild(overlay);
    document.addEventListener('mouseover', function (e) {
        const label = e.target.closest('.style_selections label');
        if (!label) return;
        label.removeEventListener("mouseout", onMouseLeave);
        label.addEventListener("mouseout", onMouseLeave);
        overlayVisible = true;
        overlay.style.opacity = "1";
        const originalText = label.querySelector("span").getAttribute("data-original-text");
        const name = originalText || label.querySelector("span").textContent;
        overlay.style.backgroundImage = `url("${samplesPath.replace(
            "fooocus_v2",
            name.toLowerCase().replaceAll(" ", "_")
        ).replaceAll("\\", "\\\\")}")`;

        tooltip.textContent = name;

        function onMouseLeave() {
            overlayVisible = false;
            overlay.style.opacity = "0";
            overlay.style.backgroundImage = "";
            label.removeEventListener("mouseout", onMouseLeave);
        }
    });
    document.addEventListener('mousemove', function (e) {
        if (!overlayVisible) return;
        overlay.style.left = `${e.clientX}px`;
        overlay.style.top = `${e.clientY}px`;
        overlay.className = e.clientY > window.innerHeight / 2 ? "lower-half" : "upper-half";
    });
}

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
    if (el === document) {
        return true;
    }

    const computedStyle = getComputedStyle(el);
    const isVisible = computedStyle.display !== 'none';

    if (!isVisible) return false;
    return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}

function playNotification() {
    gradioApp().querySelector('#audio_notification audio')?.play();
}

function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function htmlDecode(input) {
  var doc = new DOMParser().parseFromString(input, "text/html");
  return doc.documentElement.textContent;
}

/**
 * Image Library Multiselect functionality
 * 
 * Selection behavior:
 * - Click on image (not checkbox) -> Single select, show preview
 * - Click on checkbox -> Toggle multi-select for that image
 * - When multiple images selected -> Show list of selected images, no preview
 */
var librarySelectedImages = new Set();  // Stores stable image path keys
var libraryImagePaths = [];  // Maps index to serialized {path, caption}
var librarySetupTimer = null;
var libraryGalleryObserver = null;
var observedLibraryGallery = null;
var libraryBootstrapTimer = null;

function isLibraryModalOpen() {
    const modal = document.getElementById('image_library_modal');
    if (!modal) return false;
    const style = getComputedStyle(modal);
    return style.display !== 'none' && style.visibility !== 'hidden';
}

function getGradioInputElement(elemId) {
    const root = document.getElementById(elemId);
    if (!root) return null;
    if (root.tagName === 'INPUT' || root.tagName === 'TEXTAREA') return root;
    return root.querySelector('textarea, input');
}

function getGradioButtonElement(elemId) {
    const root = document.getElementById(elemId);
    if (!root) return null;
    if (root.tagName === 'BUTTON') return root;
    return root.querySelector('button');
}

function setNativeValue(input, value) {
    if (!input) return;
    const prototype = Object.getPrototypeOf(input);
    const descriptor = prototype ? Object.getOwnPropertyDescriptor(prototype, 'value') : null;
    if (descriptor && typeof descriptor.set === 'function') {
        descriptor.set.call(input, value);
    } else {
        input.value = value;
    }
}

function setNativeChecked(input, checked) {
    if (!input) return;
    const prototype = Object.getPrototypeOf(input);
    const descriptor = prototype ? Object.getOwnPropertyDescriptor(prototype, 'checked') : null;
    if (descriptor && typeof descriptor.set === 'function') {
        descriptor.set.call(input, checked);
    } else {
        input.checked = checked;
    }
}

function triggerInputEvents(input) {
    if (!input) return;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
}

function normalizeLibraryPath(value) {
    if (!value) return '';
    let normalized = String(value).trim();
    if (!normalized) return '';

    if (normalized.includes('/file=')) {
        normalized = normalized.split('/file=')[1] || normalized;
    }

    const hashIndex = normalized.indexOf('#');
    if (hashIndex >= 0) normalized = normalized.slice(0, hashIndex);
    const queryIndex = normalized.indexOf('?');
    if (queryIndex >= 0) normalized = normalized.slice(0, queryIndex);

    try {
        normalized = decodeURIComponent(normalized);
    } catch (e) {
        // Keep original if decode fails
    }
    return normalized.trim();
}

function parsePathData(pathData) {
    if (!pathData) return { path: '', caption: '', key: '' };
    try {
        const parsed = typeof pathData === 'string' ? JSON.parse(pathData) : pathData;
        const path = normalizeLibraryPath(parsed.path || '');
        const caption = normalizeLibraryPath(parsed.caption || '');
        const key = caption || path;
        return { path: path, caption: caption, key: key };
    } catch (e) {
        const fallback = normalizeLibraryPath(pathData);
        return { path: fallback, caption: '', key: fallback };
    }
}

function pruneSelectionToCurrentGallery() {
    const validKeys = new Set(
        libraryImagePaths
            .map(function(pathData) { return parsePathData(pathData).key; })
            .filter(function(key) { return !!key; })
    );

    Array.from(librarySelectedImages).forEach(function(key) {
        if (!validKeys.has(key)) {
            librarySelectedImages.delete(key);
        }
    });
}

// Initialize multiselect functionality when UI loads
onUiLoaded(function() {
    initLibraryMultiselect();
});

function initLibraryMultiselect() {
    // The gallery may render slightly after initial UI load; retry until found.
    scheduleLibrarySetup(300);
    if (libraryBootstrapTimer) {
        clearInterval(libraryBootstrapTimer);
    }
    libraryBootstrapTimer = setInterval(function() {
        const gallery = document.getElementById('library_gallery');
        if (!gallery) return;
        clearInterval(libraryBootstrapTimer);
        libraryBootstrapTimer = null;
        scheduleLibrarySetup(50);
    }, 1000);
}

function scheduleLibrarySetup(delayMs) {
    clearTimeout(librarySetupTimer);
    librarySetupTimer = setTimeout(function() {
        setupGalleryHandlers();
    }, delayMs || 80);
}

function ensureLibraryGalleryObserver(gallery) {
    if (observedLibraryGallery === gallery && libraryGalleryObserver) return;

    if (libraryGalleryObserver) {
        libraryGalleryObserver.disconnect();
    }

    observedLibraryGallery = gallery;
    libraryGalleryObserver = new MutationObserver(function(mutations) {
        if (!isLibraryModalOpen()) return;
        const hasStructuralChange = mutations.some(function(m) {
            return m.type === 'childList' && (m.addedNodes.length > 0 || m.removedNodes.length > 0);
        });
        if (hasStructuralChange) {
            scheduleLibrarySetup(60);
        }
    });
    libraryGalleryObserver.observe(gallery, { childList: true, subtree: true });
}

function setupGalleryHandlers() {
    const gallery = document.getElementById('library_gallery');
    if (!gallery) return;
    ensureLibraryGalleryObserver(gallery);
    if (!isLibraryModalOpen()) return;
    
    // Get all image paths from the gallery
    updateImagePaths(gallery);
    pruneSelectionToCurrentGallery();
    
    // Add checkbox elements and handlers to thumbnails
    const thumbnails = gallery.querySelectorAll('.thumbnail-item');
    thumbnails.forEach(function(thumb, index) {
        thumb.dataset.imageIndex = index;
        const pathData = libraryImagePaths[index] || '';
        thumb.dataset.pathData = pathData;
        const imageKey = parsePathData(pathData).key;
        
        // Add checkbox element if not already present
        let checkbox = thumb.querySelector('.library-checkbox');
        if (!checkbox) {
            checkbox = document.createElement('div');
            checkbox.className = 'library-checkbox';
            checkbox.innerHTML = '<span class="checkmark"></span>';
            thumb.appendChild(checkbox);
            
            // Handle checkbox clicks
            checkbox.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();

                const currentIndex = Number(thumb.dataset.imageIndex);
                const currentPathData = thumb.dataset.pathData || (
                    Number.isInteger(currentIndex) && currentIndex >= 0 ? (libraryImagePaths[currentIndex] || '') : ''
                );
                const currentImageKey = parsePathData(currentPathData).key;
                toggleThumbnailSelection(thumb, currentImageKey);
                
                // Communicate with Python via hidden inputs
                const pathInput = getGradioInputElement('library_checkbox_path');
                const selectedInput = getGradioInputElement('library_checkbox_selected');
                const triggerBtn = getGradioButtonElement('library_checkbox_trigger');
                
                if (pathInput && selectedInput && triggerBtn) {
                    // Update hidden inputs
                    setNativeValue(pathInput, currentPathData);
                    triggerInputEvents(pathInput);
                    setNativeChecked(selectedInput, thumb.classList.contains('selected'));
                    triggerInputEvents(selectedInput);
                    
                    // Trigger the hidden button to send data to Python
                    triggerBtn.click();
                }
                
                return false;
            }, true);
        }

        // Single click on tile switches to single-select mode, so clear multiselect state.
        if (!thumb.dataset.singleSelectHandlerAttached) {
            thumb.dataset.singleSelectHandlerAttached = '1';
            thumb.addEventListener('click', function(e) {
                if (e.target.closest('.library-checkbox')) {
                    return;
                }
                clearAllSelections();
            }, true);
        }
        
        // Restore selection state
        if (imageKey && librarySelectedImages.has(imageKey)) {
            thumb.classList.add('selected');
            checkbox.classList.add('checked');
        } else {
            thumb.classList.remove('selected');
            checkbox.classList.remove('checked');
        }
    });

    updateSelectedCountDisplay();
}

function updateImagePaths(gallery) {
    // Extract paths from gallery items
    // The gallery stores [path, caption] pairs where caption is the relative path
    libraryImagePaths = [];
    
    // Get all thumbnail items
    const items = gallery.querySelectorAll('.thumbnail-item');
    items.forEach(function(item, index) {
        // Get the img element
        const img = item.querySelector('img');
        if (img) {
            // The src is a Gradio file URL like /file=/path/to/image.png
            let src = img.src || '';
            
            // Extract path from Gradio file URL
            // Format: /file=/actual/path or /file=path
            let filePath = normalizeLibraryPath(src);
            
            // Look for the caption in the thumbnail's text content
            // Gradio gallery captions are often in a separate element
            const captionEl = item.querySelector('.caption, .gallery-item-caption, [class*="caption"]');
            let caption = captionEl ? captionEl.textContent : '';
            
            // Also check for alt text or title
            if (!caption) {
                caption = img.alt || img.title || '';
            }
            
            // Store as object with both path and caption for Python to resolve
            const pathData = JSON.stringify({
                path: filePath,
                caption: caption
            });
            libraryImagePaths[index] = pathData;
        }
    });
}

function toggleThumbnailSelection(thumb, imageKey) {
    if (!imageKey) return;
    const checkbox = thumb.querySelector('.library-checkbox');
    
    if (thumb.classList.contains('selected')) {
        thumb.classList.remove('selected');
        if (checkbox) checkbox.classList.remove('checked');
        librarySelectedImages.delete(imageKey);
    } else {
        thumb.classList.add('selected');
        if (checkbox) checkbox.classList.add('checked');
        librarySelectedImages.add(imageKey);
    }
    
    // Update the selected count display
    updateSelectedCountDisplay();
}

function updateSelectedCountDisplay() {
    const count = librarySelectedImages.size;
    const countElement = document.getElementById('library_selected_count');
    if (countElement) {
        if (count > 0) {
            countElement.innerHTML = `<span class="selected-count-badge">${count} image${count !== 1 ? 's' : ''} selected</span>`;
            countElement.style.display = 'inline-block';
        } else {
            countElement.innerHTML = '';
            countElement.style.display = 'none';
        }
    }
}

function clearAllSelections() {
    const gallery = document.getElementById('library_gallery');
    if (gallery) {
        gallery.querySelectorAll('.thumbnail-item.selected').forEach(function(thumb) {
            thumb.classList.remove('selected');
            const checkbox = thumb.querySelector('.library-checkbox');
            if (checkbox) checkbox.classList.remove('checked');
        });
    }
    librarySelectedImages.clear();
    updateSelectedCountDisplay();
}

function getSelectedPaths() {
    // Return array of paths for selected images
    return libraryImagePaths.filter(function(pathData) {
        const key = parsePathData(pathData).key;
        return key && librarySelectedImages.has(key);
    });
}

// Listen for clear selections event from Python
document.addEventListener('libraryClearSelections', function() {
    clearAllSelections();
});
