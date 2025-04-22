# from contextvars import Context
# import logging
# from logging import config
# import sys
# import os
# from pathlib import Path
# from behave import step

# # Get the absolute path to the directory containing the environment.py file (features directory)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Get the absolute path to the project's root directory (one level up)
# project_root = os.path.dirname(current_dir)
# # Get the absolute path to the src directory
# src_path = os.path.join(project_root, 'src')
# # Add the src directory to the Python path
# sys.path.insert(0, src_path)

# from automation_ocr_utils import preprocess_image, run_ocr, save_coordinates_csv, save_screenshot, load_config, init_browser, close_browser, PROJECT_ROOT
# # Configure Behave Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     handlers=[ logging.StreamHandler(sys.stdout) ] # Log to console
#     # Optional: Add FileHandler for behave logs
#     # handlers=[ logging.FileHandler(PROJECT_ROOT / "behave_run.log"), logging.StreamHandler(sys.stdout) ]
# )
# logger = logging.getLogger("behave") # Use behave's logger

# def before_all(context):
#     """Executes once before all features."""
#     logger.info("Loading configuration...")
#     try:
#         context.config = load_config()
#         # Make project root available if needed
#         context.project_root = PROJECT_ROOT
#         logger.info("Configuration loaded.")
#     except Exception as e:
#         logger.exception(f"CRITICAL: Failed to load configuration: {e}")
#         raise # Stop execution if config fails

# # def before_scenario(context, scenario):
# #     """Executes before each scenario."""
# #     logger.info(f"=== Starting Scenario: {scenario.name} ===")
# #     try:
# #         # Pass config to init_browser
# #         context.playwright, context.browser, context.browser_context, context.page = init_browser(context.config)
# #     except Exception as e:
# #         logger.exception(f"CRITICAL: Scenario setup failed (browser init): {e}")
# #         raise # Stop if browser doesn't start

# def before_scenario(context, scenario):
#     logger.info(f"=== Starting Scenario: {scenario.name} ===")
#     try:
#         context.playwright, context.browser, context.browser_context, context.page = init_browser(context.config)
        
#         def handle_navigation(page):
#             logger.info("Detected page navigation/reload. Re-running OCR...")
#             try:
#                 page.wait_for_load_state('domcontentloaded', timeout=30000)
#                 screenshot_path = save_screenshot(page, context.config, f"post_navigation_{scenario.name}")
#                 if screenshot_path:
#                     processed_image = preprocess_image(context.config, screenshot_path)
#                     if processed_image is not None:
#                         ocr_results = run_ocr(context.config, processed_image)
#                         save_coordinates_csv(context.config, ocr_results)
#                         logger.info("OCR completed and CSV updated after navigation.")
#                     else:
#                         logger.warning("Failed to preprocess image for OCR after navigation.")
#                 else:
#                     logger.warning("Failed to save screenshot for OCR after navigation.")
#             except Exception as e:
#                 logger.warning(f"Failed to re-run OCR after navigation: {e}. Proceeding without OCR update.")
        
#         context.page.on("framenavigated", lambda frame: handle_navigation(context.page) if frame == context.page.main_frame else None)
        
#     except Exception as e:
#         logger.exception(f"CRITICAL: Scenario setup failed (browser init): {e}")
#         raise

# def after_scenario(context, scenario):
#     """Executes after each scenario."""
#     logger.info(f"=== Finished Scenario: {scenario.name} - Status: {scenario.status.name} ===")
#     if hasattr(context, 'playwright'): # Check if browser was initialized
#         close_browser(context.playwright, context.browser)
#     else:
#          logger.warning("Browser resources not found for cleanup in after_scenario.")

# # def after_step(context, step):
# #     """Executes after each step."""
# #     # Take screenshot only on failure
# #     if step.status == "failed":
# #         logger.error(f"--- Step Failed: {step.keyword} {step.name} ---")
# #         if hasattr(context, 'page') and context.page and not context.page.is_closed():
# #              step_name_safe = "".join(c if c.isalnum() else "_" for c in step.name)[:80]
# #              save_screenshot(context.page, context.config, f"FAILED_{context.scenario.name}_{step_name_safe}")
# #         else:
# #              logger.warning("Could not take failure screenshot - page object not available or closed.")

# def after_step(context, step):
#     status_prefix = "FAILED" if step.status == "failed" else "PASSED" if step.status == "passed" else step.status.upper()
#     if step.status == "failed":
#         logger.error(f"--- Step Failed: {step.keyword} {step.name} ---")
#     elif step.status == "passed":
#         logger.info(f"--- Step Passed: {step.keyword} {step.name} ---")
#     else:
#         logger.warning(f"--- Step {step.status}: {step.keyword} {step.name} ---")
#     if hasattr(context, 'page') and context.page and not context.page.is_closed():
#         step_name_safe = "".join(c if c.isalnum() else "_" for c in step.name)[:80]
#         scenario_name_safe = "".join(c if c.isalnum() else "_" for c in context.scenario.name)[:80]
#         screenshot_name = f"{status_prefix}_{scenario_name_safe}_{step_name_safe}"
#         try:
#             bbox = getattr(context, 'last_element_bbox', None)
#             save_screenshot(context.page, context.config, screenshot_name, bbox=bbox)
#             logger.info(f"Saved screenshot: {screenshot_name}.png")
#         except Exception as e:
#             logger.warning(f"Failed to save screenshot '{screenshot_name}': {e}")
#     else:
#         logger.warning(f"Could not take screenshot for '{step.name}' - page object not available or closed.")

import logging
import sys
import os
from pathlib import Path
from behave import step
import time
from datetime import datetime
import traceback
from logging.handlers import RotatingFileHandler
import json
from playwright.sync_api import sync_playwright

# Calculate key directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

# Add 'src' directory to Python path
sys.path.insert(0, src_path)

from automation_ocr_utils import (
    preprocess_image,
    run_ocr,
    save_coordinates_csv,
    save_screenshot,
    load_config,
    init_browser,
    close_browser,
    PROJECT_ROOT
)

class CustomFormatter(logging.Formatter):
    """Custom formatter for detailed logging"""
    def format(self, record):
        # Add timestamp
        record.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Add context information if available
        if hasattr(record, 'context'):
            record.context_info = json.dumps(record.context, default=str)
        else:
            record.context_info = "{}"
            
        # Format the message
        return super().format(record)

def setup_logging(config):
    """Enhanced logging setup with rotation and custom formatting"""
    # Create logs directory
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Get logging configuration
    log_level = getattr(logging, config.get('Logging', 'level', fallback='INFO'))
    # Get format string and unescape %%
    log_format = config.get('Logging', 'format', fallback='%%(timestamp)s - %%(name)s - %%(levelname)s - %%(message)s').replace('%%', '%')
    log_file = config.get('Logging', 'file', fallback='automation.log')
    max_file_size = config.getint('Logging', 'max_file_size', fallback=10485760)  # 10MB
    backup_count = config.getint('Logging', 'backup_count', fallback=5)
    
    # Create logger
    logger = logging.getLogger("behave")
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = CustomFormatter(log_format)
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize global logger
logger = logging.getLogger("behave")

class AutomationError(Exception):
    """Custom exception for automation errors"""
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def retry_operation(operation, max_retries, retry_delay, max_timeout, *args, **kwargs):
    """Enhanced retry mechanism with timeout"""
    start_time = time.time()
    last_exception = None
    attempt = 0
    
    while attempt < max_retries:
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            last_exception = e
            attempt += 1
            
            # Check if we've exceeded the maximum timeout
            if time.time() - start_time > max_timeout:
                raise AutomationError(
                    f"Operation timed out after {max_timeout} seconds. Last error: {str(e)}",
                    context={"attempt": attempt, "time_elapsed": time.time() - start_time}
                )
            
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt} failed: {str(e)}. Retrying in {retry_delay} seconds...",
                    extra={"context": {"attempt": attempt, "error": str(e)}}
                )
                time.sleep(retry_delay)
    
    raise AutomationError(
        f"Operation failed after {max_retries} attempts. Last error: {str(last_exception)}",
        context={"attempts": max_retries, "last_error": str(last_exception)}
    )

def before_all(context):
    """Runs once before all features are executed."""
    try:
        context.config = load_config()
        context.project_root = PROJECT_ROOT
        context.logger = setup_logging(context.config)
        
        # Get system screen size (default to 1920x1080 if can't determine)
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except:
            screen_width = 1920
            screen_height = 1080
            
        # Store screen dimensions in context for reuse
        context.screen_size = {
            'width': screen_width,
            'height': screen_height
        }
        
        context.logger.info(f"Detected screen size: {screen_width}x{screen_height}")
        
    except Exception as e:
        logger.exception(
            "CRITICAL: Failed to initialize environment",
            extra={"context": {"error": str(e), "traceback": traceback.format_exc()}}
        )
        raise

def after_all(context):
    """Runs after all features are executed."""
    # No browser cleanup needed here as it's handled in after_scenario
    pass

def before_scenario(context, scenario):
    """Runs before each scenario starts."""
    context.logger.info(
        f"Starting Scenario: {scenario.name}",
        extra={"context": {"scenario": scenario.name, "tags": scenario.tags}}
    )
    
    try:
        # Initialize browser for each scenario
        context.playwright = sync_playwright().start()
        context.browser = context.playwright.chromium.launch(
            headless=False,
            args=['--start-maximized']
        )
        
        # Create context with stored screen size
        context.browser_context = context.browser.new_context(
            viewport=context.screen_size,
            no_viewport=True
        )
        
        # Create new page
        context.page = context.browser_context.new_page()
        
        def handle_navigation(page):
            """Enhanced navigation handler with detailed logging"""
            context.logger.info(
                "Detected page navigation/reload",
                extra={"context": {"scenario": scenario.name}}
            )
            
            try:
                page.wait_for_load_state('domcontentloaded')
                page.wait_for_timeout(2000)  # Wait for 2 seconds after load
                
                screenshot_path = save_screenshot(page, context.config, f"post_navigation_{scenario.name}")
                if screenshot_path:
                    processed_image = preprocess_image(context.config, screenshot_path)
                    if processed_image is not None:
                        ocr_results = run_ocr(context.config, processed_image)
                        if ocr_results:
                            save_coordinates_csv(context.config, ocr_results)
                            context.logger.info("OCR completed and CSV updated")
                            return
                context.logger.warning("Failed to process navigation OCR")
            except Exception as e:
                context.logger.warning(f"Navigation handler error: {str(e)}")

        context.page.on(
            "framenavigated",
            lambda frame: handle_navigation(context.page) if frame == context.page.main_frame else None
        )
        
        context.logger.info("Browser initialized successfully for scenario")
        
    except Exception as e:
        context.logger.exception(
            "Scenario setup failed",
            extra={"context": {
                "scenario": scenario.name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }}
        )
        raise

def after_scenario(context, scenario):
    """Runs after each scenario completes."""
    try:
        if hasattr(context, 'page') and context.page:
            context.page.close()
        if hasattr(context, 'browser_context') and context.browser_context:
            context.browser_context.close()
        if hasattr(context, 'browser') and context.browser:
            context.browser.close()
        if hasattr(context, 'playwright') and context.playwright:
            context.playwright.stop()
        
        context.logger.info("Browser resources cleaned up successfully")
    except Exception as e:
        context.logger.error(f"Error during browser cleanup: {str(e)}")

def after_step(context, step):
    """Runs after each step within a scenario."""
    status_prefix = "FAILED" if step.status == "failed" else "PASSED" if step.status == "passed" else step.status.upper()
    
    log_context = {
        "step": step.name,
        "keyword": step.keyword,
        "status": step.status,
        "duration": step.duration if hasattr(step, 'duration') else None
    }
    
    if step.status == "failed":
        context.logger.error(
            f"Step Failed: {step.keyword} {step.name}",
            extra={"context": {**log_context, "error": str(step.exception) if hasattr(step, 'exception') else None}}
        )
    elif step.status == "passed":
        context.logger.info(
            f"Step Passed: {step.keyword} {step.name}",
            extra={"context": log_context}
        )
    else:
        context.logger.warning(
            f"Step {step.status}: {step.keyword} {step.name}",
            extra={"context": log_context}
        )
    
    if hasattr(context, 'page') and context.page and not context.page.is_closed():
        step_name_safe = "".join(c for c in step.name if c.isalnum() or c in ('_', '-'))[:80]
        scenario_name_safe = "".join(c for c in context.scenario.name if c.isalnum() or c in ('_', '-'))[:80]
        screenshot_name = f"{status_prefix}_{scenario_name_safe}_{step_name_safe}"
        
        try:
            bbox = getattr(context, 'last_element_bbox', None)
            screenshot_path = save_screenshot(context.page, context.config, screenshot_name, bbox=bbox)
            context.logger.info(
                "Saved screenshot",
                extra={"context": {
                    "screenshot": str(screenshot_path),
                    "bbox": bbox
                }}
            )
        except Exception as e:
            context.logger.warning(
                "Failed to save screenshot",
                extra={"context": {
                    "screenshot_name": screenshot_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }}
            )
    else:
        context.logger.warning(
            "Could not take screenshot",
            extra={"context": {
                "step": step.name,
                "reason": "page object not available or closed"
            }}
        )