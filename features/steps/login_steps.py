import logging
from behave import *
from playwright.sync_api import expect # Using playwright's assertions
from behave import given  # You might want to use @when instead of @given for this step
import time
from configparser import ConfigParser
import difflib

# Import helper functions
# Assuming environment.py added 'src' to sys.path
from automation_ocr_utils import (
    drag_and_drop_coordinates, save_screenshot, preprocess_image, run_ocr, save_coordinates_csv,
    load_coordinates_csv, find_element_by_label,
    click_coordinates, fill_coordinates, slide_to_position, find_elements_by_label,
    get_coordinates_from_csv
)

logger = logging.getLogger(__name__)

# Use type hints for context for better IDE support
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from behave.runner import Context
    from configparser import ConfigParser
    from playwright.sync_api import Page

@given('I navigate to "{url}" and perform OCR')
@when('I navigate to "{url}" and perform OCR')
def step_navigate_and_ocr(context: 'Context', url: str):
    """Navigates and performs the initial OCR scan."""
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Navigating to: {url}")
    timeout = config.getint('Playwright', 'navigation_timeout', fallback=60000)

    def attempt_navigation():
        try:
            page.goto(url, timeout=timeout, wait_until='domcontentloaded')
            logger.info(f"Navigation to {url} succeeded.")
            return True
        except Exception as e:
            logger.warning(f"Navigation attempt failed: {e}")
            return False

    max_attempts = 3
    for attempt in range(max_attempts):
        if attempt_navigation():
            break
        logger.info(f"Retrying navigation to {url} (Attempt {attempt + 2}/{max_attempts})...")
        page.wait_for_timeout(2000)
        if attempt == max_attempts - 1:
            logger.error(f"Failed to navigate to {url} after {max_attempts} attempts.")
            raise Exception(f"Navigation to {url} failed after {max_attempts} attempts.")

    try:
        logger.info("Performing initial OCR scan...")
        screenshot_path = save_screenshot(page, config, f"initial_navigation_{url.split('/')[-1]}")
        if screenshot_path:
            processed_image = preprocess_image(config, screenshot_path)
            if processed_image is not None:
                ocr_results = run_ocr(config, processed_image)
                save_coordinates_csv(config, ocr_results)
            else:
                logger.warning("Failed to preprocess screenshot for OCR.")
        else:
            logger.warning("Failed to save screenshot for OCR.")
    except Exception as e:
        logger.error(f"Failed during navigation or initial OCR for {url}: {e}")
        raise

def scroll_to_element(context, bbox, element_type='element', confidence=None):
    """Scroll to make an element visible with enhanced visual feedback and debugging tools."""
    try:
        # Get viewport dimensions
        viewport_height = context.page.viewport_size['height']
        viewport_width = context.page.viewport_size['width']
        
        # Calculate element center
        element_center_x = bbox['X'] + (bbox['Width'] / 2)
        element_center_y = bbox['Y'] + (bbox['Height'] / 2)
        
        # Calculate scroll position to center the element
        scroll_x = element_center_x - (viewport_width / 2)
        scroll_y = element_center_y - (viewport_height / 2)
        
        # Add enhanced visual feedback and debugging styles
        context.page.evaluate("""
            () => {
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes pulse {
                        0% { transform: scale(1); opacity: 0.3; }
                        50% { transform: scale(1.05); opacity: 0.6; }
                        100% { transform: scale(1); opacity: 0.3; }
                    }
                    @keyframes fadeIn {
                        from { opacity: 0; }
                        to { opacity: 1; }
                    }
                    @keyframes fadeOut {
                        from { opacity: 1; }
                        to { opacity: 0; }
                    }
                    @keyframes success {
                        0% { transform: scale(1); }
                        50% { transform: scale(1.2); }
                        100% { transform: scale(1); }
                    }
                    .automation-highlight {
                        position: absolute;
                        border: 3px solid #4CAF50;
                        background-color: rgba(76, 175, 80, 0.1);
                        z-index: 9999;
                        pointer-events: none;
                        animation: pulse 2s infinite, fadeIn 0.3s ease-out;
                    }
                    .automation-highlight::before {
                        content: '';
                        position: absolute;
                        top: -10px;
                        left: -10px;
                        right: -10px;
                        bottom: -10px;
                        border: 2px dashed #4CAF50;
                        animation: pulse 2s infinite;
                        opacity: 0.5;
                    }
                    .automation-highlight::after {
                        content: '↓';
                        position: absolute;
                        top: -30px;
                        left: 50%;
                        transform: translateX(-50%);
                        color: #4CAF50;
                        font-size: 24px;
                        font-weight: bold;
                        animation: fadeIn 0.3s ease-out;
                    }
                    .automation-debug {
                        position: absolute;
                        background: rgba(0, 0, 0, 0.8);
                        color: white;
                        padding: 5px;
                        font-family: monospace;
                        font-size: 12px;
                        border-radius: 3px;
                        z-index: 10000;
                        pointer-events: none;
                    }
                    .automation-confidence {
                        position: absolute;
                        right: -5px;
                        top: -5px;
                        background: #FF5722;
                        color: white;
                        padding: 2px 5px;
                        border-radius: 3px;
                        font-size: 10px;
                        z-index: 10001;
                    }
                    .automation-focus-ring {
                        position: absolute;
                        border: 2px solid #2196F3;
                        border-radius: 3px;
                        pointer-events: none;
                        z-index: 9998;
                    }
                    .automation-success {
                        animation: success 0.5s ease-out;
                        border-color: #4CAF50 !important;
                    }
                    .automation-error {
                        border-color: #F44336 !important;
                        animation: pulse 1s infinite;
                    }
                    .automation-aria-label {
                        position: absolute;
                        background: rgba(0, 0, 0, 0.8);
                        color: white;
                        padding: 5px;
                        border-radius: 3px;
                        font-size: 12px;
                        z-index: 10002;
                        pointer-events: none;
                        white-space: nowrap;
                    }
                `;
                document.head.appendChild(style);
            }
        """)
        
        # Add the highlight element with enhanced features
        context.page.evaluate("""
            (bbox, elementType, confidence) => {
                const highlight = document.createElement('div');
                highlight.className = 'automation-highlight';
                highlight.style.left = bbox.X + 'px';
                highlight.style.top = bbox.Y + 'px';
                highlight.style.width = bbox.Width + 'px';
                highlight.style.height = bbox.Height + 'px';
                highlight.id = 'automation-highlight';
                
                // Add debug information
                const debug = document.createElement('div');
                debug.className = 'automation-debug';
                debug.style.left = bbox.X + 'px';
                debug.style.top = (bbox.Y - 20) + 'px';
                debug.textContent = `X: ${bbox.X}, Y: ${bbox.Y}, W: ${bbox.Width}, H: ${bbox.Height}`;
                debug.id = 'automation-debug';
                
                // Add confidence indicator if available
                if (confidence !== null) {
                    const conf = document.createElement('div');
                    conf.className = 'automation-confidence';
                    conf.textContent = `${Math.round(confidence * 100)}%`;
                    highlight.appendChild(conf);
                }
                
                // Add focus ring for keyboard navigation
                const focusRing = document.createElement('div');
                focusRing.className = 'automation-focus-ring';
                focusRing.style.left = (bbox.X - 2) + 'px';
                focusRing.style.top = (bbox.Y - 2) + 'px';
                focusRing.style.width = (bbox.Width + 4) + 'px';
                focusRing.style.height = (bbox.Height + 4) + 'px';
                focusRing.id = 'automation-focus-ring';
                
                // Add ARIA label
                const ariaLabel = document.createElement('div');
                ariaLabel.className = 'automation-aria-label';
                ariaLabel.style.left = bbox.X + 'px';
                ariaLabel.style.top = (bbox.Y + bbox.Height + 5) + 'px';
                ariaLabel.textContent = `${elementType} element`;
                ariaLabel.id = 'automation-aria-label';
                
                document.body.appendChild(highlight);
                document.body.appendChild(debug);
                document.body.appendChild(focusRing);
                document.body.appendChild(ariaLabel);
            }
        """, bbox, element_type, confidence)
        
        # Smooth scroll with enhanced feedback
        steps = 20
        for i in range(steps):
            t = i / steps
            ease = 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
            
            current_scroll_x = (scroll_x * ease)
            current_scroll_y = (scroll_y * ease)
            context.page.mouse.wheel(current_scroll_x, current_scroll_y)
            context.page.wait_for_timeout(25)
        
        # Wait for scroll to complete
        context.page.wait_for_timeout(300)
        
        # Add interaction feedback based on element type
        context.page.evaluate("""
            (elementType) => {
                const highlight = document.getElementById('automation-highlight');
                if (highlight) {
                    let feedbackText = '';
                    let color = '#4CAF50';
                    
                    switch(elementType) {
                        case 'text input':
                            feedbackText = '✎';
                            color = '#2196F3';
                            break;
                        case 'button':
                            feedbackText = '↖';
                            color = '#FF9800';
                            break;
                        case 'slider':
                            feedbackText = '↔';
                            color = '#9C27B0';
                            break;
                        case 'checkbox':
                            feedbackText = '✓';
                            color = '#E91E63';
                            break;
                        case 'radio button':
                            feedbackText = '●';
                            color = '#00BCD4';
                            break;
                        default:
                            feedbackText = '↓';
                    }
                    
                    highlight.style.borderColor = color;
                    highlight.style.backgroundColor = color.replace(')', ', 0.1)').replace('rgb', 'rgba');
                    highlight.setAttribute('data-feedback', feedbackText);
                    
                    // Update ARIA label
                    const ariaLabel = document.getElementById('automation-aria-label');
                    if (ariaLabel) {
                        ariaLabel.textContent = `${elementType} element - ${feedbackText}`;
                    }
                }
            }
        """, element_type)
        
        # Show success state briefly
        context.page.evaluate("""
            () => {
                const highlight = document.getElementById('automation-highlight');
                if (highlight) {
                    highlight.classList.add('automation-success');
                }
            }
        """)
        
        # Keep elements visible for a bit longer
        context.page.wait_for_timeout(1500)
        
        # Remove elements with fade out
        context.page.evaluate("""
            () => {
                const elements = [
                    'automation-highlight',
                    'automation-debug',
                    'automation-focus-ring',
                    'automation-aria-label'
                ];
                
                elements.forEach(id => {
                    const element = document.getElementById(id);
                    if (element) {
                        element.style.animation = 'fadeOut 0.5s ease-out';
                        setTimeout(() => element.remove(), 500);
                    }
                });
            }
        """)
        
        logger.info(f"Scrolled to {element_type} at position ({element_center_x}, {element_center_y})")
        return True
    except Exception as e:
        # Show error state
        context.page.evaluate("""
            () => {
                const highlight = document.getElementById('automation-highlight');
                if (highlight) {
                    highlight.classList.add('automation-error');
                }
            }
        """)
        logger.warning(f"Error during scroll to {element_type}: {str(e)}")
        return False

@when('I enter "{value}" using label "{target_label}"')
def step_enter_text(context: 'Context', value: str, target_label: str):
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Entering '{value}' using label '{target_label}'")
    try:
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            raise Exception("No OCR results loaded from CSV to find label.")
        target_element = find_element_by_label(config, target_label, ocr_results, element_type='text', context=context)
        if not target_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}")
            raise Exception(f"Label '{target_label}' not found in OCR results with sufficient confidence.")
        
        # Scroll to element with highlighting
        scroll_to_element(context, target_element, 'text input')
        
        context.last_element_bbox = {'X': target_element['X'], 'Y': target_element['Y'], 'Width': target_element['Width'], 'Height': target_element['Height']}
        
        # Click the element to focus it
        click_coordinates(page, target_element, context=context)
        
        # Fill the value directly without any keyboard shortcuts
        fill_coordinates(page, target_element, value, context=context, clear_first=False)
        
        logger.info(f"Successfully entered '{value}' for label '{target_label}'.")
    except Exception as e:
        logger.exception(f"Failed to enter '{value}' for label '{target_label}': {e}")
        raise

@when('I click using label "{target_label}"')
@step('I click using label "{target_label}"')
def step_click_by_label(context: 'Context', target_label: str):
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Attempting to click element labeled '{target_label}'")
    try:
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            raise Exception("No OCR results loaded from CSV to find label.")
        target_element = find_element_by_label(config, target_label, ocr_results, element_type='button', context=context, last_interacted_bbox=getattr(context, 'last_element_bbox', None))
        if not target_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}")
            raise Exception(f"Label '{target_label}' not found in OCR results with sufficient confidence.")
        
        # Scroll to element with highlighting
        scroll_to_element(context, target_element, 'button')
        
        context.last_element_bbox = {'X': target_element['X'], 'Y': target_element['Y'], 'Width': target_element['Width'], 'Height': target_element['Height'], 'Label': target_label}
        click_coordinates(page, target_element, context=context)
        logger.info(f"Successfully clicked label '{target_label}'.")
        page.wait_for_load_state('networkidle', timeout=10000)
        logger.info("Performing OCR scan after click...")
        screenshot_path = save_screenshot(page, config, f"after_click_{target_label}")
        if screenshot_path:
            processed_image = preprocess_image(config, screenshot_path)
            if processed_image is not None:
                ocr_results = run_ocr(config, processed_image)
                save_coordinates_csv(config, ocr_results)
    except Exception as e:
        logger.exception(f"Failed to click label '{target_label}': {e}")
        raise

@when(u'I see the label "{target_label}"')
def step_see_label(context: 'Context', target_label: str):
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Checking for label '{target_label}' on the page...")
    try:
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            logger.warning("No OCR results in CSV. Re-running OCR...")
            screenshot_path = save_screenshot(page, config, f"before_see_label_{target_label}")
            if screenshot_path:
                processed_image = preprocess_image(config, screenshot_path)
                if processed_image is not None:
                    ocr_results = run_ocr(config, processed_image)
                    save_coordinates_csv(config, ocr_results)
                else:
                    raise Exception("Failed to preprocess screenshot for OCR.")
            else:
                raise Exception("Failed to save screenshot for OCR.")

        target_element = find_element_by_label(config, target_label, ocr_results, element_type='label', context=context)
        if target_element is None:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}")
            raise AssertionError(f"Label '{target_label}' was not found in OCR results.")
        context.last_element_bbox = {'X': target_element['X'], 'Y': target_element['Y'], 'Width': target_element['Width'], 'Height': target_element['Height']}
        logger.info(f"Label '{target_label}' found: '{target_element['Label']}' at X={target_element['X']}, Y={target_element['Y']}")
    except Exception as e:
        logger.exception(f"Failed to verify label '{target_label}': {e}")
        raise

@when('I wait for {seconds} seconds')
def step_wait(context: 'Context', seconds: str):
    """Waits for the specified number of seconds."""
    logger.info(f"Waiting for {seconds} seconds...")
    try:
        page: 'Page' = context.page
        page.wait_for_timeout(float(seconds) * 1000)
        logger.info(f"Waited for {seconds} seconds.")
    except Exception as e:
        logger.error(f"Failed to wait for {seconds} seconds: {e}")
        raise

@when('I slide the slider labeled "{target_label}" to position {target_x:d}')
def step_slide_slider_to_position(context: 'Context', target_label: str, target_x: int):
    """Slides a slider element identified by label to a target x coordinate."""
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Attempting to slide slider labeled '{target_label}' to position {target_x}")
    try:
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            raise Exception("No OCR results loaded from CSV to find label.")
        slider_element = find_element_by_label(config, target_label, ocr_results, element_type='slider', context=context)
        if not slider_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}")
            raise Exception(f"Slider label '{target_label}' not found in OCR results with sufficient confidence.")
        
        # Scroll to element with highlighting
        scroll_to_element(context, slider_element, 'slider')
        
        context.last_element_bbox = {'X': slider_element['X'], 'Y': slider_element['Y'], 'Width': slider_element['Width'], 'Height': slider_element['Height'], 'Label': target_label}
        slide_to_position(page, slider_element, target_x, context=context)
        logger.info(f"Successfully slid slider '{target_label}' to position {target_x}.")
        # Take full page screenshot after sliding
        screenshot_path = save_screenshot(page, config, f"after_slide_{target_label}_to_{target_x}")
        if screenshot_path:
            logger.info(f"Screenshot saved after sliding slider '{target_label}' to position {target_x}: {screenshot_path}")
    except Exception as e:
        logger.exception(f"Failed to slide slider '{target_label}' to position {target_x}: {e}")
        raise

@when('I check the checkbox using label "{target_label}"')
def step_check_checkbox(context: 'Context', target_label: str):
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Attempting to check checkbox using label '{target_label}'")
    try:
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            raise Exception("No OCR results loaded from CSV to find label.")
        target_element = find_element_by_label(config, target_label, ocr_results, element_type='checkbox', context=context)
        if not target_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}")
            raise Exception(f"Label '{target_label}' not found in OCR results with sufficient confidence.")
        
        # Scroll to element with highlighting
        scroll_to_element(context, target_element, 'checkbox')
        
        context.last_element_bbox = {'X': target_element['X'], 'Y': target_element['Y'], 'Width': target_element['Width'], 'Height': target_element['Height']}
        click_coordinates(page, target_element, context=context)
        logger.info(f"Successfully checked checkbox for label '{target_label}'.")
    except Exception as e:
        logger.exception(f"Failed to check checkbox for label '{target_label}': {e}")
        raise

@when('I select the radio button using label "{target_label}"')
def step_select_radio(context: 'Context', target_label: str):
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Attempting to select radio button using label '{target_label}'")
    try:
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            raise Exception("No OCR results loaded from CSV to find label.")
        target_element = find_element_by_label(config, target_label, ocr_results, element_type='radio', context=context)
        if not target_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}")
            raise Exception(f"Label '{target_label}' not found in OCR results with sufficient confidence.")
        
        # Scroll to element with highlighting
        scroll_to_element(context, target_element, 'radio button')
        
        context.last_element_bbox = {'X': target_element['X'], 'Y': target_element['Y'], 'Width': target_element['Width'], 'Height': target_element['Height']}
        click_coordinates(page, target_element, context=context)
        logger.info(f"Successfully selected radio button for label '{target_label}'.")
    except Exception as e:
        logger.exception(f"Failed to select radio button for label '{target_label}': {e}")
        raise

@when(u'I drag element labeled "{source_label}" and drop on "{target_label}"')
def step_drag_and_drop(context, source_label, target_label):
    """Drag and drop an element from source to target using OCR-detected coordinates."""
    logger.info(f"Attempting to drag element labeled '{source_label}' and drop on '{target_label}'")
    
    # Get coordinates for source and target elements
    ocr_results = load_coordinates_csv(context.config)
    if not ocr_results:
        logger.warning("No OCR results found. Re-running OCR...")
        # Take a new screenshot and run OCR
        screenshot_path = save_screenshot(context.page, context.config, "retry_ocr_drag_drop")
        if screenshot_path:
            processed_image = preprocess_image(context.config, screenshot_path)
            if processed_image is not None:
                ocr_results = run_ocr(context.config, processed_image)
                save_coordinates_csv(context.config, ocr_results)
                logger.info("OCR re-run completed with new results")
            else:
                raise ValueError("Failed to preprocess image for OCR retry")
        else:
            raise ValueError("Failed to save screenshot for OCR retry")
    
    # Try to find source element with more lenient matching
    source_bbox = None
    target_bbox = None
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # Find source element
            source_bbox = find_element_by_label(context.config, source_label, ocr_results, element_type='drag')
            if not source_bbox:
                # Try fuzzy matching if exact match fails
                source_matches = find_elements_by_label(context.config, source_label, ocr_results)
                if source_matches:
                    source_bbox = source_matches[0]  # Use first match
                    logger.warning(f"Using fuzzy match for source element: {source_bbox['Label']}")
            
            # Find target element
            target_bbox = find_element_by_label(context.config, target_label, ocr_results, element_type='drop')
            if not target_bbox:
                # Try fuzzy matching if exact match fails
                target_matches = find_elements_by_label(context.config, target_label, ocr_results)
                if target_matches:
                    target_bbox = target_matches[0]  # Use first match
                    logger.warning(f"Using fuzzy match for target element: {target_bbox['Label']}")
            
            if source_bbox and target_bbox:
                break  # Both elements found, exit retry loop
            
            if attempt < max_attempts - 1:
                logger.info(f"Retry attempt {attempt + 1}/{max_attempts} - Re-running OCR...")
                # Take a new screenshot and run OCR
                screenshot_path = save_screenshot(context.page, context.config, f"retry_ocr_attempt_{attempt + 1}")
                if screenshot_path:
                    processed_image = preprocess_image(context.config, screenshot_path)
                    if processed_image is not None:
                        ocr_results = run_ocr(context.config, processed_image)
                        save_coordinates_csv(context.config, ocr_results)
                        logger.info(f"OCR re-run completed for attempt {attempt + 1}")
                    else:
                        logger.warning(f"Failed to preprocess image for attempt {attempt + 1}")
                else:
                    logger.warning(f"Failed to save screenshot for attempt {attempt + 1}")
                
                # Wait a bit before retrying
                context.page.wait_for_timeout(1000)
        
        except Exception as e:
            logger.warning(f"Error during attempt {attempt + 1}: {str(e)}")
            if attempt == max_attempts - 1:
                raise
    
    # Validate elements were found
    if not source_bbox:
        save_screenshot(context.page, context.config, f"source_not_found_{source_label}")
        raise ValueError(f"Could not find source element with label '{source_label}' after {max_attempts} attempts. Please check the screenshot for OCR results.")
    
    if not target_bbox:
        save_screenshot(context.page, context.config, f"target_not_found_{target_label}")
        raise ValueError(f"Could not find target element with label '{target_label}' after {max_attempts} attempts. Please check the screenshot for OCR results.")
    
    # Scroll to make both elements visible
    try:
        # Calculate the center point between source and target
        source_center_y = source_bbox['Y'] + (source_bbox['Height'] / 2)
        target_center_y = target_bbox['Y'] + (target_bbox['Height'] / 2)
        center_y = (source_center_y + target_center_y) / 2
        
        # Get viewport height
        viewport_height = context.page.viewport_size['height']
        
        # Calculate scroll position to center both elements
        scroll_y = center_y - (viewport_height / 2)
        
        # Scroll to position
        context.page.mouse.wheel(0, scroll_y)
        logger.info(f"Scrolled to position {scroll_y} to make both elements visible")
        
        # Wait for scroll to complete
        context.page.wait_for_timeout(500)
        
        # Take screenshot after scrolling
        save_screenshot(context.page, context.config, f"after_scroll_{source_label}_to_{target_label}")
    except Exception as e:
        logger.warning(f"Error during scroll: {str(e)}")
        # Continue with drag and drop even if scroll fails
    
    # Save screenshot before drag and drop
    save_screenshot(context.page, context.config, f"before_drag_{source_label}_to_{target_label}")
    
    # Perform drag and drop with retry
    for attempt in range(max_attempts):
        try:
            drag_and_drop_coordinates(context.page, source_bbox, target_bbox)
            logger.info(f"Successfully dragged '{source_label}' to '{target_label}'")
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                save_screenshot(context.page, context.config, f"drag_drop_error_{source_label}_to_{target_label}")
                raise Exception(f"Failed to perform drag and drop after {max_attempts} attempts: {str(e)}")
            logger.warning(f"Drag and drop attempt {attempt + 1} failed: {str(e)}")
            context.page.wait_for_timeout(1000)
    
    # Take screenshot after drag and drop
    screenshot_path = save_screenshot(context.page, context.config, f"after_drag_{source_label}_to_{target_label}")
    if screenshot_path:
        logger.info(f"Screenshot saved after drag and drop: {screenshot_path}")

@when('I click using label "{target_label}" and perform OCR')
def step_click_and_perform_ocr(context: 'Context', target_label: str):
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Attempting to click element labeled '{target_label}' and perform OCR")
    try:
        # Load existing OCR results
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            logger.warning("No OCR results loaded from CSV. Re-running initial OCR...")
            screenshot_path = save_screenshot(page, config, f"before_click_{target_label}")
            if screenshot_path:
                processed_image = preprocess_image(config, screenshot_path)
                if processed_image is not None:
                    ocr_results = run_ocr(config, processed_image)
                    save_coordinates_csv(config, ocr_results)
                else:
                    raise Exception("Failed to preprocess screenshot for initial OCR.")
            else:
                raise Exception("Failed to save screenshot for initial OCR.")

        # Find the target element
        target_element = find_element_by_label(config, target_label, ocr_results, element_type='button', context=context)
        if not target_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}")
            raise Exception(f"Label '{target_label}' not found in OCR results with sufficient confidence.")

        # Set bbox for highlighting before click
        context.last_element_bbox = {'X': target_element['X'], 'Y': target_element['Y'], 'Width': target_element['Width'], 'Height': target_element['Height']}
        save_screenshot(page, config, f"before_click_{target_label}")

        # Perform the click
        click_coordinates(page, target_element, context=context)
        logger.info(f"Successfully clicked label '{target_label}'.")

        # Wait for potential navigation and re-run OCR
        try:
            page.wait_for_load_state('domcontentloaded', timeout=10000)
            logger.info("Page load detected after click, re-running OCR...")
            screenshot_path = save_screenshot(page, config, f"after_click_{target_label}")
            if screenshot_path:
                processed_image = preprocess_image(config, screenshot_path)
                if processed_image is not None:
                    ocr_results = run_ocr(config, processed_image)
                    save_coordinates_csv(config, ocr_results)
                    logger.info("OCR updated successfully after click.")
                else:
                    logger.warning("Failed to preprocess image after click.")
            else:
                logger.warning("Failed to save screenshot after click.")
        except Exception as e:
            logger.warning(f"No navigation detected after click: {e}. Forcing OCR re-run...")
            screenshot_path = save_screenshot(page, config, f"force_ocr_after_click_{target_label}")
            if screenshot_path:
                processed_image = preprocess_image(config, screenshot_path)
                if processed_image is not None:
                    ocr_results = run_ocr(config, processed_image)
                    save_coordinates_csv(config, ocr_results)
                    logger.info("Forced OCR update completed.")
                else:
                    logger.warning("Forced OCR preprocess failed.")

        # Update bbox for highlighting after OCR (optional, based on new results)
        if ocr_results:
            target_element = find_element_by_label(config, target_label, ocr_results, element_type='button', context=context)
            if target_element:
                context.last_element_bbox = {'X': target_element['X'], 'Y': target_element['Y'], 'Width': target_element['Width'], 'Height': target_element['Height']}
                logger.debug(f"Updated bbox after OCR: {context.last_element_bbox}")

    except Exception as e:
        logger.exception(f"Failed to click label '{target_label}' and perform OCR: {e}")
        raise

@given(u'I wait for 5 seconds for the page to load')
def step_impl(context):
    """Wait for the page to load and stabilize."""
    try:
        context.logger.info("Waiting for page to load and stabilize...")
        context.page.wait_for_load_state('domcontentloaded')
        context.page.wait_for_timeout(5000)  # 5 seconds
        context.logger.info("Page load wait completed")
    except Exception as e:
        context.logger.error(f"Error during page load wait: {str(e)}")
        raise

@then(u'I wait for 10 seconds')
def step_impl(context):
    """Wait for a specified duration."""
    try:
        context.logger.info("Waiting for 10 seconds...")
        context.page.wait_for_timeout(10000)  # 10 seconds
        context.logger.info("Wait completed")
    except Exception as e:
        context.logger.error(f"Error during wait: {str(e)}")
        raise

@then(u'I see the label "{label_text}"')
def step_impl(context, label_text):
    """Verify that a specific label is visible on the page."""
    try:
        context.logger.info(f"Verifying label: {label_text}")
        
        # Get coordinates from CSV
        coordinates = get_coordinates_from_csv(context.config, label_text)
        if not coordinates:
            # Take a screenshot for debugging
            save_screenshot(context.page, context.config, f"label_not_found_{label_text}")
            raise AssertionError(f"Label '{label_text}' not found in coordinates CSV")
        
        # Take a screenshot for verification
        screenshot_path = save_screenshot(context.page, context.config, f"verify_label_{label_text}")
        if not screenshot_path:
            raise AssertionError("Failed to save verification screenshot")
        
        # Perform OCR on the screenshot
        processed_image = preprocess_image(context.config, screenshot_path)
        if processed_image is None:
            raise AssertionError("Failed to preprocess image for verification")
        
        ocr_results = run_ocr(context.config, processed_image)
        if not ocr_results:
            raise AssertionError("No OCR results found for verification")
        
        # Check if the label text is found in OCR results
        label_found = any(
            difflib.SequenceMatcher(None, result['text'].lower(), label_text.lower()).ratio() 
            >= context.config.getfloat('Matching', 'label_match_threshold', fallback=0.6)
            for result in ocr_results
        )
        
        if not label_found:
            raise AssertionError(f"Label '{label_text}' not found on page")
        
        context.logger.info(f"Label '{label_text}' verified successfully")
        
    except Exception as e:
        context.logger.error(f"Error verifying label: {str(e)}")
        raise