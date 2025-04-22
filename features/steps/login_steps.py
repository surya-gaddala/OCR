import logging
from behave import *
from playwright.sync_api import expect # Using playwright's assertions
from behave import given  # You might want to use @when instead of @given for this step
import time
from configparser import ConfigParser
import difflib
from playwright.sync_api import sync_playwright
from coordinate_automation import CoordinateAutomation
import os
from datetime import datetime

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
    timeout = config.getint('Playwright', 'navigation_timeout', fallback=30000)  # Reduced from 60000

    def attempt_navigation():
        try:
            page.goto(url, timeout=timeout, wait_until='domcontentloaded')
            logger.info(f"Navigation to {url} succeeded.")
            return True
        except Exception as e:
            logger.warning(f"Navigation attempt failed: {e}")
            return False

    max_attempts = 2  # Reduced from 3
    for attempt in range(max_attempts):
        if attempt_navigation():
            break
        logger.info(f"Retrying navigation to {url} (Attempt {attempt + 2}/{max_attempts})...")
        page.wait_for_timeout(1000)  # Reduced from 2000
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
        
        # First try to find scrollable container
        scrollable_container = context.page.evaluate("""
            () => {
                // Common selectors for scrollable containers in ordering systems
                const selectors = [
                    'div[class*="modal"]',
                    'div[class*="content"]',
                    'div[class*="container"]',
                    'div[class*="main"]',
                    'div[class*="scroll"]'
                ];
                
                for (const selector of selectors) {
                    const elements = Array.from(document.querySelectorAll(selector));
                    const scrollable = elements.find(el => {
                        const style = window.getComputedStyle(el);
                        const overflow = style.overflow + style.overflowY;
                        return overflow.includes('scroll') || overflow.includes('auto');
                    });
                    if (scrollable) return scrollable;
                }
                return null;
            }
        """)
        
        # Add enhanced visual feedback and debugging styles
        context.page.evaluate("""
            () => {
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes pulse {
                        0% { transform: scale(1); opacity: 0.6; }
                        50% { transform: scale(1.02); opacity: 0.8; }
                        100% { transform: scale(1); opacity: 0.6; }
                    }
                    @keyframes glow {
                        0% { box-shadow: 0 0 5px rgba(255, 0, 0, 0.5); }
                        50% { box-shadow: 0 0 20px rgba(255, 0, 0, 0.8); }
                        100% { box-shadow: 0 0 5px rgba(255, 0, 0, 0.5); }
                    }
                    .automation-highlight {
                        position: absolute;
                        border: 3px solid #FF0000;
                        background-color: rgba(255, 0, 0, 0.1);
                        z-index: 9999;
                        pointer-events: none;
                        animation: pulse 2s infinite, glow 2s infinite;
                    }
                    .automation-target {
                        position: absolute;
                        width: 20px;
                        height: 20px;
                        border-radius: 50%;
                        background-color: rgba(255, 0, 0, 0.7);
                        transform: translate(-50%, -50%);
                        z-index: 10000;
                        pointer-events: none;
                        animation: pulse 1s infinite;
                    }
                    .automation-label {
                        position: absolute;
                        background-color: rgba(0, 0, 0, 0.8);
                        color: white;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        z-index: 10001;
                        pointer-events: none;
                        white-space: nowrap;
                    }
                `;
                document.head.appendChild(style);
            }
        """)
        
        # Add visual highlight with operation type label
        highlight_elements = context.page.evaluate("""
            (bbox, elementType) => {
                // Create highlight box
                const highlight = document.createElement('div');
                highlight.className = 'automation-highlight';
                highlight.style.left = bbox.X + 'px';
                highlight.style.top = bbox.Y + 'px';
                highlight.style.width = bbox.Width + 'px';
                highlight.style.height = bbox.Height + 'px';
                
                // Create target indicator
                const target = document.createElement('div');
                target.className = 'automation-target';
                target.style.left = (bbox.X + bbox.Width / 2) + 'px';
                target.style.top = (bbox.Y + bbox.Height / 2) + 'px';
                
                // Create operation label
                const label = document.createElement('div');
                label.className = 'automation-label';
                label.style.left = (bbox.X + bbox.Width / 2) + 'px';
                label.style.top = (bbox.Y - 25) + 'px';
                label.style.transform = 'translateX(-50%)';
                
                // Set label text based on element type
                switch(elementType) {
                    case 'text input':
                        label.textContent = '✎ Text Input';
                        break;
                    case 'button':
                        label.textContent = '👆 Click';
                        break;
                    case 'radio button':
                        label.textContent = '⚪ Radio Select';
                        break;
                    case 'checkbox':
                        label.textContent = '☐ Checkbox';
                        break;
                    case 'slider':
                        label.textContent = '↔ Slider';
                        break;
                    default:
                        label.textContent = 'Interact';
                }
                
                document.body.appendChild(highlight);
                document.body.appendChild(target);
                document.body.appendChild(label);
                
                return [highlight.id, target.id, label.id];
            }
        """, bbox, element_type)
        
        # Perform scroll
        if scrollable_container:
            logger.info("Found scrollable container, scrolling within container")
            context.page.evaluate("""
                (container, elementY) => {
                    const containerRect = container.getBoundingClientRect();
                    const scrollTop = elementY - containerRect.height / 3;
                    container.scrollTo({
                        top: scrollTop,
                        behavior: 'smooth'
                    });
                }
            """, scrollable_container, element_center_y)
        else:
            logger.info("No scrollable container found, using window scroll")
            # Calculate scroll position to center the element
            scroll_y = element_center_y - (viewport_height / 3)  # Position element at 1/3 from top
            
            # Smooth scroll
            steps = 20
            current_scroll = context.page.evaluate("() => window.pageYOffset")
            scroll_distance = scroll_y - current_scroll
            
            for i in range(steps):
                progress = i / steps
                ease = 2 * progress * progress if progress < 0.5 else -1 + (4 - 2 * progress) * progress
                current_step = current_scroll + (scroll_distance * ease)
                context.page.evaluate(f"window.scrollTo(0, {current_step})")
                context.page.wait_for_timeout(25)
        
        # Wait for scroll to settle
        context.page.wait_for_timeout(500)
        
        # Keep highlight visible for interaction
        context.page.wait_for_timeout(2000)
        
        # Remove highlights with fade out
        context.page.evaluate("""
            (elementIds) => {
                elementIds.forEach(id => {
                    const element = document.getElementById(id);
                    if (element) {
                        element.style.animation = 'fadeOut 0.5s ease-out';
                        setTimeout(() => element.remove(), 500);
                    }
                });
            }
        """, highlight_elements)
        
        logger.info(f"Scrolled to {element_type} at position ({element_center_x}, {element_center_y})")
        return True
        
    except Exception as e:
        logger.warning(f"Error during scroll to {element_type}: {str(e)}")
        return False

@when('I enter "{value}" using label "{target_label}"')
def step_enter_text(context: 'Context', value: str, target_label: str):
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Entering '{value}' using label '{target_label}'")
    try:
        # Take screenshot before entering text
        save_screenshot(page, config, f"before_enter_{target_label}")
        
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
        page.wait_for_timeout(500)  # Wait for focus
        
        # Clear existing text if any
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        page.wait_for_timeout(500)
        
        # Fill the value
        fill_coordinates(page, target_element, value, context=context, clear_first=False)
        page.wait_for_timeout(1000)  # Wait for text to be entered
        
        # Take screenshot after entering text
        save_screenshot(page, config, f"after_enter_{target_label}")
        
        # Verify the text was entered
        page.wait_for_timeout(1000)
        logger.info(f"Successfully entered '{value}' for label '{target_label}'")
    except Exception as e:
        logger.exception(f"Failed to enter '{value}' for label '{target_label}': {e}")
        raise

@when('I click using label "{target_label}" and perform OCR')
@step('I click using label "{target_label}" and perform OCR')
def step_click_and_perform_ocr(context: 'Context', target_label: str):
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Attempting to click element labeled '{target_label}' and perform OCR")
    
    try:
        # Load existing OCR results
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            logger.warning("No OCR results found, taking new screenshot and running OCR...")
            screenshot_path = save_screenshot(page, config, f"before_click_{target_label}")
            if screenshot_path:
                processed_image = preprocess_image(config, screenshot_path)
                if processed_image is not None:
                    ocr_results = run_ocr(config, processed_image)
                    save_coordinates_csv(config, ocr_results)
                else:
                    raise Exception("Failed to preprocess image for OCR")
            else:
                raise Exception("Failed to save screenshot for OCR")

        # Try to find the target element with more lenient matching
        target_element = find_element_by_label(config, target_label, ocr_results, element_type='button', context=context)
        
        if not target_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}")
            raise Exception(f"Label '{target_label}' not found in OCR results with sufficient confidence.")
        
        # Scroll to element with highlighting
        scroll_to_element(context, target_element, 'button')
        
        # Set bbox for highlighting before click
        context.last_element_bbox = {
            'X': target_element['X'],
            'Y': target_element['Y'],
            'Width': target_element['Width'],
            'Height': target_element['Height']
        }

        # Perform the click
        click_coordinates(page, target_element, context=context)
        logger.info(f"Successfully clicked label '{target_label}'.")

        # Wait for potential navigation and re-run OCR only if needed
        try:
            page.wait_for_load_state('networkidle', timeout=5000)  # Reduced from 10000
            # Only re-run OCR if the page actually changed
            if page.url != context.last_url:
                logger.info("Page changed, re-running OCR...")
                screenshot_path = save_screenshot(page, config, f"after_click_{target_label}")
                if screenshot_path:
                    processed_image = preprocess_image(config, screenshot_path)
                    if processed_image is not None:
                        ocr_results = run_ocr(config, processed_image)
                        save_coordinates_csv(config, ocr_results)
                        logger.info("OCR updated successfully after click.")
                context.last_url = page.url
        except Exception as e:
            logger.warning(f"No navigation detected after click: {e}")

    except Exception as e:
        logger.exception(f"Failed to click label '{target_label}' and perform OCR: {e}")
        save_screenshot(page, config, f"error_click_{target_label}")
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
        # Take screenshot before selection
        save_screenshot(page, config, f"before_radio_{target_label}")
        
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            raise Exception("No OCR results loaded from CSV to find label.")
            
        # Find the radio button element
        target_element = find_element_by_label(config, target_label, ocr_results, element_type='radio', context=context)
        if not target_element:
            save_screenshot(page, config, f"LABEL_NOT_FOUND_{target_label}")
            raise Exception(f"Label '{target_label}' not found in OCR results with sufficient confidence.")
        
        # First, scroll the label into view with some padding
        viewport_height = page.viewport_size['height']
        element_y = target_element['Y']
        
        # Calculate scroll position to center the element with padding
        scroll_y = max(0, element_y - (viewport_height / 3))  # Position element at 1/3 from top
        
        # Smooth scroll to the element
        logger.info(f"Scrolling to radio button label at Y={element_y}")
        steps = 20
        current_scroll = page.evaluate("() => window.pageYOffset")
        scroll_distance = scroll_y - current_scroll
        
        for i in range(steps):
            progress = i / steps
            ease = 2 * progress * progress if progress < 0.5 else -1 + (4 - 2 * progress) * progress
            current_step = current_scroll + (scroll_distance * ease)
            page.evaluate(f"window.scrollTo(0, {current_step})")
            page.wait_for_timeout(25)
        
        # Wait for scroll to settle
        page.wait_for_timeout(500)
        
        # Add visual highlight for the radio button area
        scroll_to_element(context, target_element, 'radio button')
        
        # Calculate click coordinates - adjust slightly to the left of the label
        # This helps ensure we click the actual radio input element
        adjusted_x = target_element['X'] - 20  # Move 20 pixels left of the label
        adjusted_y = target_element['Y'] + (target_element['Height'] / 2)
        
        # Try clicking up to 3 times with verification
        max_attempts = 3
        for attempt in range(max_attempts):
            logger.info(f"Radio button click attempt {attempt + 1}/{max_attempts}")
            
            # Add visual feedback before click
            page.evaluate("""
                (x, y) => {
                    const dot = document.createElement('div');
                    dot.style.position = 'absolute';
                    dot.style.left = (x - 5) + 'px';
                    dot.style.top = (y - 5) + 'px';
                    dot.style.width = '10px';
                    dot.style.height = '10px';
                    dot.style.backgroundColor = 'red';
                    dot.style.borderRadius = '50%';
                    dot.style.zIndex = '9999';
                    dot.style.boxShadow = '0 0 10px rgba(255, 0, 0, 0.5)';
                    document.body.appendChild(dot);
                    setTimeout(() => dot.remove(), 1000);
                }
            """, adjusted_x, adjusted_y)
            
            # Ensure we're still scrolled to the right position before clicking
            current_scroll = page.evaluate("() => window.pageYOffset")
            if abs(current_scroll - scroll_y) > 50:  # If we've drifted more than 50px
                logger.info("Readjusting scroll position before click")
                page.evaluate(f"window.scrollTo(0, {scroll_y})")
                page.wait_for_timeout(300)
            
            # Click the radio button
            page.mouse.click(adjusted_x, adjusted_y)
            page.wait_for_timeout(500)  # Wait for any animations
            
            # Take verification screenshot
            screenshot_path = save_screenshot(page, config, f"verify_radio_{target_label}_attempt_{attempt + 1}")
            
            # Wait for potential state change
            page.wait_for_timeout(1000)
            
            # Try to verify if radio is selected using JavaScript
            try:
                is_checked = page.evaluate("""
                    (x, y) => {
                        const element = document.elementFromPoint(x, y);
                        if (element && element.type === 'radio') {
                            return element.checked;
                        }
                        // Try to find nearby radio input
                        const radio = document.querySelector(`input[type="radio"][name="${element?.name}"]`);
                        return radio ? radio.checked : false;
                    }
                """, adjusted_x, adjusted_y)
                
                if is_checked:
                    logger.info(f"Radio button successfully selected on attempt {attempt + 1}")
                    break
                else:
                    logger.warning(f"Radio button not verified as selected on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Failed to verify radio button state: {e}")
            
            if attempt < max_attempts - 1:
                page.wait_for_timeout(500)  # Wait before next attempt
        
        # Final verification screenshot
        save_screenshot(page, config, f"after_radio_{target_label}")
        
        # Update last interacted element
        context.last_element_bbox = {
            'X': adjusted_x,
            'Y': adjusted_y,
            'Width': target_element['Width'],
            'Height': target_element['Height']
        }
        
        logger.info(f"Completed radio button selection process for '{target_label}'")
        
    except Exception as e:
        logger.exception(f"Failed to select radio button for label '{target_label}': {e}")
        # Take error screenshot
        save_screenshot(page, config, f"error_radio_{target_label}")
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
@given(u'I see the label "{label_text}"')
def verify_label(context: 'Context', label_text: str):
    """Verify that a specific label is visible on the page.
    This step can be used with both Given and Then keywords.
    """
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Verifying label: {label_text}")
    
    try:
        # Load existing OCR results
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            logger.warning("No OCR results found, taking new screenshot and running OCR...")
            screenshot_path = save_screenshot(page, config, f"verify_label_{label_text}")
            if screenshot_path:
                processed_image = preprocess_image(config, screenshot_path)
                if processed_image is not None:
                    ocr_results = run_ocr(config, processed_image)
                    save_coordinates_csv(config, ocr_results)
                else:
                    raise Exception("Failed to preprocess image for OCR")
            else:
                raise Exception("Failed to save screenshot for OCR")

        # Find the label in OCR results
        target_element = find_element_by_label(config, label_text, ocr_results, element_type='text', context=context)
        if not target_element:
            # Try fuzzy matching if exact match fails
            matches = find_elements_by_label(config, label_text, ocr_results)
            if matches:
                target_element = matches[0]  # Use first match
                logger.info(f"Found label '{label_text}' using fuzzy matching")
            else:
                save_screenshot(page, config, f"label_not_found_{label_text}")
                raise AssertionError(f"Label '{label_text}' not found on page")

        # Scroll to element for visual verification
        scroll_to_element(context, target_element, 'text')
        
        # Take verification screenshot
        save_screenshot(page, config, f"verified_label_{label_text}")
        
        logger.info(f"Successfully verified label: {label_text}")
        
    except Exception as e:
        logger.error(f"Failed to verify label '{label_text}': {str(e)}")
        save_screenshot(page, config, f"error_verify_label_{label_text}")
        raise AssertionError(f"Failed to verify label '{label_text}': {str(e)}")

@when('I select "{option}" from dropdown labeled "{dropdown_label}"')
def step_select_from_dropdown(context: 'Context', option: str, dropdown_label: str):
    """Select an option from a dropdown menu using OCR-detected coordinates."""
    page: 'Page' = context.page
    config: ConfigParser = context.config
    logger.info(f"Attempting to select '{option}' from dropdown labeled '{dropdown_label}'")
    
    try:
        # Load existing OCR results
        ocr_results = load_coordinates_csv(config)
        if not ocr_results:
            logger.warning("No OCR results found, taking new screenshot and running OCR...")
            screenshot_path = save_screenshot(page, config, f"before_dropdown_{dropdown_label}")
            if screenshot_path:
                processed_image = preprocess_image(config, screenshot_path)
                if processed_image is not None:
                    ocr_results = run_ocr(config, processed_image)
                    save_coordinates_csv(config, ocr_results)
                else:
                    raise Exception("Failed to preprocess image for OCR")
            else:
                raise Exception("Failed to save screenshot for OCR")

        # First find the dropdown label
        dropdown_element = find_element_by_label(config, dropdown_label, ocr_results, element_type='text', context=context)
        if not dropdown_element:
            # Try fuzzy matching
            matches = find_elements_by_label(config, dropdown_label, ocr_results)
            if matches:
                dropdown_element = matches[0]
                logger.info(f"Found dropdown label '{dropdown_label}' using fuzzy matching")
            else:
                save_screenshot(page, config, f"dropdown_not_found_{dropdown_label}")
                raise AssertionError(f"Dropdown label '{dropdown_label}' not found on page")

        # Scroll to dropdown
        scroll_to_element(context, dropdown_element, 'dropdown')
        
        # Click the dropdown to open it
        click_coordinates(page, dropdown_element, context=context)
        page.wait_for_timeout(500)  # Reduced from 1000
        
        # Find and click the option
        option_element = find_element_by_label(config, option, ocr_results, element_type='text', context=context)
        if not option_element:
            # Try fuzzy matching for option
            matches = find_elements_by_label(config, option, ocr_results)
            if matches:
                option_element = matches[0]
                logger.info(f"Found option '{option}' using fuzzy matching")
            else:
                save_screenshot(page, config, f"option_not_found_{option}")
                raise AssertionError(f"Option '{option}' not found in dropdown")
        
        # Click the option
        click_coordinates(page, option_element, context=context)
        logger.info(f"Selected option '{option}' from dropdown '{dropdown_label}'")
        
        # Wait for any animations to complete
        page.wait_for_timeout(500)  # Reduced from 1000

    except Exception as e:
        logger.exception(f"Failed to select '{option}' from dropdown '{dropdown_label}': {e}")
        save_screenshot(page, config, f"error_dropdown_{dropdown_label}_{option}")
        raise

def before_scenario(context, scenario):
    """Initialize page for each scenario using existing browser context."""
    try:
        # Create new page in existing browser context
        context.page = context.browser_context.new_page()
        context.automation = CoordinateAutomation(context.page)
        context.last_url = None  # Track last URL for change detection
        logger.info("New page created for scenario")
    except Exception as e:
        logger.error(f"Failed to initialize page for scenario: {e}")
        raise

def after_scenario(context, scenario):
    """Cleanup after each scenario."""
    try:
        # Close only the page, not the browser
        if hasattr(context, 'page'):
            context.page.close()
        logger.info("Page closed successfully")
    except Exception as e:
        logger.error(f"Failed to cleanup after scenario: {e}")
        raise

@given('I am on the login page')
def step_impl(context):
    context.automation.navigate("https://example.com/login")

@when('I enter username "{username}"')
def step_impl(context, username):
    context.automation.type("Username", username)

@when('I enter password "{password}"')
def step_impl(context, password):
    context.automation.type("Password", password)

@when('I click the "{button}" button')
def step_impl(context, button):
    context.automation.click(button)
    context.automation.check_page_change()

@then('I should be redirected to the dashboard')
def step_impl(context):
    # Add verification logic here
    context.automation.check_page_change()

@given('I maximize the browser window')
def step_maximize_window(context: 'Context'):
    """Maximize the browser window."""
    try:
        page: 'Page' = context.page
        logger.info("Maximizing browser window")
        page.set_viewport_size({"width": 1920, "height": 1080})  # Full HD resolution
        context.browser.windows()[0].maximize()
        logger.info("Browser window maximized successfully")
    except Exception as e:
        logger.error(f"Failed to maximize browser window: {e}")
        raise