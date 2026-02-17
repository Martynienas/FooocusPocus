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
var librarySelectedImages = new Set();  // Stores image paths
var libraryImagePaths = [];  // Maps index to path

// Initialize multiselect functionality when UI loads
onUiLoaded(function() {
    initLibraryMultiselect();
});

function initLibraryMultiselect() {
    // Watch for the library gallery to be added/modified
    const observer = new MutationObserver(function(mutations) {
        const gallery = document.getElementById('library_gallery');
        if (gallery) {
            setupGalleryHandlers();
        }
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Initial setup
    setTimeout(setupGalleryHandlers, 500);
}

function setupGalleryHandlers() {
    const gallery = document.getElementById('library_gallery');
    if (!gallery) return;
    
    // Get all image paths from the gallery
    updateImagePaths(gallery);
    
    // Add click handlers to thumbnails
    const thumbnails = gallery.querySelectorAll('.thumbnail-item');
    thumbnails.forEach(function(thumb, index) {
        // Remove any existing handlers
        thumb.removeEventListener('click', handleThumbnailClick);
        thumb.addEventListener('click', handleThumbnailClick);
        
        // Set data attribute for index
        thumb.dataset.imageIndex = index;
    });
}

function updateImagePaths(gallery) {
    // Extract paths from gallery items
    libraryImagePaths = [];
    const items = gallery.querySelectorAll('.thumbnail-item img');
    items.forEach(function(img, index) {
        // Get the src and extract the path
        const src = img.src;
        // The path is usually in the src as a file path or data attribute
        // We'll store the src for now and let Python resolve it
        libraryImagePaths[index] = src;
    });
}

function handleThumbnailClick(e) {
    const thumb = e.target.closest('.thumbnail-item');
    if (!thumb) return;
    
    const index = parseInt(thumb.dataset.imageIndex, 10);
    const rect = thumb.getBoundingClientRect();
    
    // Check if click is in the checkbox area (top-right corner, 20x20px)
    const checkboxArea = {
        left: rect.right - 25,
        right: rect.right - 5,
        top: rect.top + 5,
        bottom: rect.top + 25
    };
    
    const isCheckboxClick = (
        e.clientX >= checkboxArea.left &&
        e.clientX <= checkboxArea.right &&
        e.clientY >= checkboxArea.top &&
        e.clientY <= checkboxArea.bottom
    );
    
    if (isCheckboxClick) {
        // Toggle selection
        e.preventDefault();
        e.stopPropagation();
        toggleThumbnailSelection(thumb, index);
        
        // Communicate with Python via hidden inputs
        const pathInput = document.getElementById('library_checkbox_path');
        const selectedInput = document.getElementById('library_checkbox_selected');
        const triggerBtn = document.getElementById('library_checkbox_trigger');
        
        if (pathInput && selectedInput && triggerBtn) {
            // Get the image path from the thumbnail
            const img = thumb.querySelector('img');
            const imagePath = img ? img.src : libraryImagePaths[index];
            
            // Update hidden inputs
            pathInput.value = imagePath;
            selectedInput.checked = thumb.classList.contains('selected');
            
            // Trigger the hidden button to send data to Python
            triggerBtn.click();
        }
    }
    // For non-checkbox clicks, let the default Gradio gallery select handler work
    // This will show the preview
}

function toggleThumbnailSelection(thumb, index) {
    if (thumb.classList.contains('selected')) {
        thumb.classList.remove('selected');
        librarySelectedImages.delete(index);
    } else {
        thumb.classList.add('selected');
        librarySelectedImages.add(index);
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
        });
    }
    librarySelectedImages.clear();
    updateSelectedCountDisplay();
}

function getSelectedPaths() {
    // Return array of paths for selected images
    return Array.from(librarySelectedImages).map(function(index) {
        return libraryImagePaths[index];
    }).filter(function(path) {
        return path !== undefined;
    });
}

// Re-apply selections when gallery content changes
onUiUpdate(function(mutations) {
    const gallery = document.getElementById('library_gallery');
    if (gallery) {
        updateImagePaths(gallery);
        
        // Re-apply selection state to thumbnails
        setTimeout(function() {
            const thumbnails = gallery.querySelectorAll('.thumbnail-item');
            thumbnails.forEach(function(thumb, index) {
                thumb.dataset.imageIndex = index;
                thumb.removeEventListener('click', handleThumbnailClick);
                thumb.addEventListener('click', handleThumbnailClick);
                
                // Restore selection state
                if (librarySelectedImages.has(index)) {
                    thumb.classList.add('selected');
                }
            });
        }, 100);
    }
});

// Listen for clear selections event from Python
document.addEventListener('libraryClearSelections', function() {
    clearAllSelections();
});