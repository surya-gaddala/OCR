import configparser
import logging
import csv
import re
import difflib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from playwright.sync_api import sync_playwright
from fuzzywuzzy import fuzz
import math
import os
import json
import hashlib

# OCR and Image Processing Libs
import cv2
import numpy as np
import pytesseract
import easyocr

# Browser Libs
from playwright.sync_api import sync_playwright, Page, Playwright, Browser, BrowserContext, Error as PlaywrightError

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / 'config' / 'config.ini'

# --- Configuration ---

def load_config() -> configparser.ConfigParser:
    """Loads config and sets Tesseract path."""
    if not CONFIG_FILE.exists(): raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    tesseract_path_str = config.get('OCR', 'tesseract_path', fallback=None)
    tesseract_found = False
    if tesseract_path_str:
        tesseract_exe = Path(tesseract_path_str)
        if tesseract_exe.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_exe)
            logger.info(f"Using Tesseract from config: {tesseract_exe}")
            tesseract_found = True
        else:
            logger.warning(f"Tesseract path from config not found: {tesseract_exe}.")
    if not tesseract_found: # Check PATH if not found via config
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract found in system PATH.")
            tesseract_found = True
        except pytesseract.TesseractNotFoundError:
            pass # Logged later if actually needed by engine choice
    if not tesseract_found and config.get('OCR', 'ocr_engine', fallback='combined').lower() in ['tesseract', 'combined']:
        logger.error("Tesseract selected as OCR engine but not found. Check PATH or config.ini.")
        raise RuntimeError("Tesseract not found.")
    return config

# --- Browser Functions ---

def init_browser(config) -> tuple[Playwright, Browser, BrowserContext, Page]:
    """Initialize Playwright browser instance."""
    headless = config.getboolean('Playwright', 'headless', fallback=False)
    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(no_viewport=True)
        page = context.new_page()
        if not headless:
            try:
                page.evaluate("() => { window.moveTo(0, 0); window.resizeTo(screen.width, screen.height); }")
            except PlaywrightError as e:
                logger.warning(f"Could not maximize window via JS: {e}")
        logger.info(f"Browser initialized (headless={headless})")
        return playwright, browser, context, page
    except Exception as e:
        logger.error(f"Browser init failed: {e}"); raise

def close_browser(playwright: Playwright | None, browser: Browser | None):
    """Safely close Playwright resources."""
    logger.info("Closing browser resources...")
    try:
        if browser and browser.is_connected(): browser.close()
        if playwright: playwright.stop()
        logger.info("Browser closed.")
    except Exception as e: logger.error(f"Error closing browser: {e}")

def save_screenshot(page: Page, config, name_prefix: str, bbox: Optional[Dict] = None) -> Optional[Path]:
    """Saves a full-page screenshot or a screenshot of a specific bounding box.
    Updated to avoid cropping screenshots as per user request."""
    screenshot_dir_str = config.get('General', 'screenshot_dir', fallback='screenshots')
    screenshot_dir = PROJECT_ROOT / screenshot_dir_str
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{name_prefix}_{timestamp}.png"
    path = screenshot_dir / filename
    try:
        # Always save full page screenshot, ignore bbox cropping
        page.screenshot(path=path, full_page=True)
        logger.info(f"Full page screenshot saved: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to save screenshot {path}: {e}"); return None

# --- Image Preprocessing ---

# Global OCR reader instances with lazy loading
_tesseract_reader = None
_easyocr_reader = None
_ocr_cache = {}
_preprocessed_cache = {}

def get_ocr_reader(engine: str = 'easyocr') -> Any:
    """Get or initialize the OCR reader instance with optimized settings."""
    global _tesseract_reader, _easyocr_reader
    
    if engine.lower() == 'tesseract':
        if _tesseract_reader is None:
            logger.info("Initializing Tesseract OCR reader...")
            _tesseract_reader = pytesseract
            logger.info("Tesseract OCR Reader initialized")
        return _tesseract_reader
    else:
        if _easyocr_reader is None:
            logger.info("Initializing EasyOCR reader...")
            # Enable GPU if available, fallback to CPU
            try:
                _easyocr_reader = easyocr.Reader(['en'], gpu=True)
                logger.info("EasyOCR Reader initialized with GPU support")
            except Exception as e:
                logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
                _easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR Reader initialized with CPU")
        return _easyocr_reader

def get_image_hash(image: np.ndarray) -> str:
    """Generate a hash for an image to use as cache key."""
    # Use a faster hashing method for large images
    if image.size > 1000000:  # For large images
        resized = cv2.resize(image, (100, 100))
        return hashlib.md5(resized.tobytes()).hexdigest()
    return hashlib.md5(image.tobytes()).hexdigest()

def preprocess_image(config: Any, image_path: str) -> Optional[np.ndarray]:
    """Optimize image preprocessing with enhanced error handling."""
    try:
        # Check if we have a cached preprocessed image
        image_hash = hashlib.md5(str(image_path).encode()).hexdigest()
        if image_hash in _preprocessed_cache:
            logger.debug("Using cached preprocessed image")
            return _preprocessed_cache[image_hash]

        # Read image with validation
        if not Path(image_path).exists():
            logger.error(f"Image file not found: {image_path}")
            return None
            
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Resize large images to improve performance
        max_dimension = 2000
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # Enhanced preprocessing with error handling
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better text detection
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Denoise to remove small artifacts
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Cache the preprocessed image
            _preprocessed_cache[image_hash] = denoised
            
            # Clear cache if it gets too large
            if len(_preprocessed_cache) > 100:
                logger.info("Clearing preprocessing cache to prevent memory issues")
                _preprocessed_cache.clear()
                
            return denoised
            
        except Exception as proc_error:
            logger.error(f"Image preprocessing failed: {proc_error}")
            # Fallback to simple thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        return None

# --- OCR Execution ---

def run_tesseract(image: np.ndarray, config) -> List[Dict]:
    results = []; min_conf_perc = config.getfloat('OCR', 'confidence_threshold', fallback=0.4) * 100
    try: data = pytesseract.image_to_data(image, config=r'--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError: logger.error("Tesseract not found."); raise
    except Exception as e: logger.error(f"Tesseract execution error: {e}"); return results
    for i in range(len(data['level'])):
        text = data['text'][i].strip(); conf = int(data['conf'][i])
        if conf >= min_conf_perc and text:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            if w > 5 and h > 5:
                results.append({'Label': text, 'X': x, 'Y': y, 'Width': w, 'Height': h, 'Confidence': round(conf / 100.0, 4), 'Engine': 'Tesseract'})
    logger.debug(f"Tesseract found {len(results)} results meeting confidence.")
    return results

def run_easyocr(image: np.ndarray, config) -> List[Dict]:
    global easyocr_reader;
    if easyocr_reader is None: initialize_easyocr(config)
    if easyocr_reader == "init_failed": return []
    results = []; min_conf = config.getfloat('OCR', 'confidence_threshold', fallback=0.4)
    try: ocr_output = easyocr_reader.readtext(image, detail=1, paragraph=False)
    except Exception as e: logger.error(f"EasyOCR execution error: {e}"); return results
    for (bbox, text, conf) in ocr_output:
        text = text.strip()
        if conf >= min_conf and text:
            x_coords = [int(point[0]) for point in bbox]; y_coords = [int(point[1]) for point in bbox]
            x, y = min(x_coords), min(y_coords); w = max(x_coords) - x; h = max(y_coords) - y
            if w > 5 and h > 5: results.append({'Label': text, 'X': x, 'Y': y, 'Width': w, 'Height': h, 'Confidence': round(conf, 4), 'Engine': 'EasyOCR'})
    logger.debug(f"EasyOCR found {len(results)} results meeting confidence.")
    return results

def run_ocr(config: Any, image: np.ndarray) -> List[Dict[str, Any]]:
    """Run OCR with enhanced reliability and error handling."""
    try:
        # Validate input
        if image is None or image.size == 0:
            logger.error("Invalid image input")
            return []
            
        # Generate image hash for cache key
        image_hash = get_image_hash(image)
        
        # Check cache first
        if image_hash in _ocr_cache:
            logger.debug("Using cached OCR results")
            return _ocr_cache[image_hash]

        # Get appropriate OCR reader based on config
        engine = config.get('OCR', 'ocr_engine', fallback='easyocr').lower()
        reader = get_ocr_reader(engine)
        
        # Run OCR with enhanced error handling
        formatted_results = []
        try:
            if engine == 'tesseract':
                # Use Tesseract with optimized settings
                custom_config = r'--oem 3 --psm 6'
                results = reader.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
                
                for i in range(len(results['text'])):
                    try:
                        text = results['text'][i].strip()
                        conf = float(results['conf'][i])
                        
                        if conf > 30 and text:  # Confidence threshold and non-empty text
                            formatted_results.append({
                                'text': text,
                                'confidence': conf / 100,
                                'X': int(results['left'][i]),
                                'Y': int(results['top'][i]),
                                'Width': int(results['width'][i]),
                                'Height': int(results['height'][i])
                            })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping invalid Tesseract result: {e}")
                        continue
                        
            else:
                # Use EasyOCR with optimized settings
                results = reader.readtext(image, paragraph=False)
                
                for (bbox, text, conf) in results:
                    try:
                        text = text.strip()
                        if conf > 0.3 and text:  # Confidence threshold and non-empty text
                            x1, y1 = bbox[0]
                            x3, y3 = bbox[2]
                            formatted_results.append({
                                'text': text,
                                'confidence': float(conf),
                                'X': int(x1),
                                'Y': int(y1),
                                'Width': int(x3 - x1),
                                'Height': int(y3 - y1)
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid EasyOCR result: {e}")
                        continue
                        
        except Exception as ocr_error:
            logger.error(f"OCR processing failed: {ocr_error}")
            return []
            
        # Log detected text for debugging
        if formatted_results:
            logger.debug(f"Detected {len(formatted_results)} text items:")
            for result in formatted_results:
                logger.debug(f"Text: '{result['text']}' at ({result['X']}, {result['Y']}) with confidence {result['confidence']}")
        else:
            logger.warning("No text detected in image")

        # Cache the results
        _ocr_cache[image_hash] = formatted_results
        
        # Limit cache size to prevent memory issues
        if len(_ocr_cache) > 1000:
            _ocr_cache.clear()
            logger.info("Cleared OCR cache to prevent memory issues")
            
        return formatted_results

    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return []

# --- Coordinate/Label Handling ---

def save_coordinates_csv(config: Any, coordinates: List[Dict[str, Any]]) -> None:
    """Save OCR coordinates to CSV with robust error handling."""
    try:
        output_file = Path('detected_coordinates.csv')
        
        # Ensure coordinates is not empty
        if not coordinates:
            logger.warning("No coordinates to save")
            return
            
        # Create backup of existing file if it exists
        if output_file.exists():
            backup_file = output_file.with_suffix('.csv.bak')
            try:
                output_file.rename(backup_file)
                logger.info(f"Created backup of existing CSV: {backup_file}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        # Write new CSV with proper error handling
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['Label', 'X', 'Y', 'Width', 'Height', 'Confidence']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for coord in coordinates:
                try:
                    # Ensure all required fields are present with defaults
                    row = {
                        'Label': str(coord.get('text', '')).strip(),
                        'X': int(coord.get('X', 0)),
                        'Y': int(coord.get('Y', 0)),
                        'Width': int(coord.get('Width', 0)),
                        'Height': int(coord.get('Height', 0)),
                        'Confidence': float(coord.get('confidence', 0))
                    }
                    writer.writerow(row)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid coordinate entry: {e}")
                    continue
                    
        logger.info(f"Successfully saved {len(coordinates)} coordinates to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save coordinates: {str(e)}")
        # Restore backup if available
        if 'backup_file' in locals() and backup_file.exists():
            try:
                backup_file.rename(output_file)
                logger.info("Restored backup CSV file")
            except Exception as restore_error:
                logger.error(f"Failed to restore backup: {restore_error}")

def load_coordinates_csv(config) -> List[Dict]:
    """Loads coordinates from the runtime CSV file."""
    output_csv_str = config.get('General', 'runtime_ocr_csv', fallback='detected_coordinates.csv')
    input_path = PROJECT_ROOT / output_csv_str
    if not input_path.exists(): logger.warning(f"Runtime OCR CSV not found: {input_path}"); return []
    results = []
    try:
        with open(input_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames or []
            required_fields = ['Label', 'X', 'Y', 'Width', 'Height', 'Confidence']
            if not all(f in fieldnames for f in required_fields):
                logger.error(f"CSV file {input_path} missing required columns. Expected: {required_fields}")
                return []
            for row in reader:
                try: # Add type conversion and validation
                    results.append({
                        'Label': row['Label'],
                        'X': int(float(row['X'])), 'Y': int(float(row['Y'])),
                        'Width': int(float(row['Width'])), 'Height': int(float(row['Height'])),
                        'Confidence': float(row['Confidence']),
                        'Engine': row.get('Engine', '') # Optional field
                    })
                except (ValueError, KeyError) as conv_err:
                    logger.warning(f"Skipping row due to conversion error: {row} - {conv_err}")
        logger.debug(f"Loaded {len(results)} coordinates from {input_path}")
        return results
    except Exception as e: logger.error(f"Failed to load runtime CSV {input_path}: {e}"); return []

def normalize_text(text: str) -> str:
    """Normalize text for matching (lowercase, alphanumeric, strip trailing punctuation like colon)."""
    if not text:
        return ''
    text = text.lower().strip()
    # Remove trailing punctuation like colon, period, comma
    text = re.sub(r'[:.,;]+$', '', text)
    # Remove all non-alphanumeric characters
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def find_element_by_label(config: configparser.ConfigParser, target_label: str, ocr_results: List[Dict], element_type: str = 'button', context=None, last_interacted_bbox: Optional[Dict] = None) -> Optional[Dict]:
    """Finds an element by label using OCR results with enhanced normalization and fallback to fuzzy matching.
    Logs all candidate matches and warns on ambiguity."""
    if not ocr_results:
        logger.warning("No OCR results provided for label matching.")
        return None

    # Special handling for button text variations
    button_text_variations = {
        'Add to Cart': ['Add to Cart', 'Add to cart', 'Add to cart', 'Add to Cart', 'Add to cart', 'Add to Cart'],
        'Shopping Cart': ['Shopping Cart', 'Shopping cart', 'Shopping cart', 'Shopping Cart'],
        'Checkout': ['Checkout', 'Check out', 'Check Out'],
        'Login': ['Login', 'Log in', 'Log In']
    }

    # Get variations for the target label if it's a known button
    variations = button_text_variations.get(target_label, [target_label])
    logger.info(f"Looking for variations of '{target_label}': {variations}")

    norm_target = normalize_text(target_label)
    norm_variations = [normalize_text(v) for v in variations]

    def calculate_distance(bbox1, bbox2):
        """Calculate Euclidean distance between center points of two bounding boxes."""
        cx1 = bbox1['X'] + bbox1['Width'] / 2
        cy1 = bbox1['Y'] + bbox1['Height'] / 2
        cx2 = bbox2['X'] + bbox2['Width'] / 2
        cy2 = bbox2['Y'] + bbox2['Height'] / 2
        return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    def is_same_element(bbox1, bbox2, threshold=0.3):
        """Check if two bounding boxes represent the same element using multiple criteria."""
        # Calculate overlap
        x_overlap = max(0, min(bbox1['X'] + bbox1['Width'], bbox2['X'] + bbox2['Width']) - max(bbox1['X'], bbox2['X']))
        y_overlap = max(0, min(bbox1['Y'] + bbox1['Height'], bbox2['Y'] + bbox2['Height']) - max(bbox1['Y'], bbox2['Y']))
        overlap_area = x_overlap * y_overlap
        
        # Calculate areas
        area1 = bbox1['Width'] * bbox1['Height']
        area2 = bbox2['Width'] * bbox2['Height']
        
        # Calculate overlap ratio
        overlap_ratio = overlap_area / min(area1, area2)
        
        # Calculate center distance
        center_distance = calculate_distance(bbox1, bbox2)
        max_dimension = max(bbox1['Width'], bbox1['Height'], bbox2['Width'], bbox2['Height'])
        normalized_distance = center_distance / max_dimension
        
        # Consider both overlap and distance
        return overlap_ratio > threshold or normalized_distance < 0.2

    def filter_duplicates(matches: List[Dict]) -> List[Dict]:
        """Filter out duplicate matches using multiple criteria."""
        if not matches:
            return matches
            
        # Sort by confidence, size, and position
        matches.sort(key=lambda m: (
            -m.get('Confidence', 0),  # Higher confidence first
            -(m['Width'] * m['Height']),  # Larger elements first
            m['Y'],  # Higher up on the page first
            m['X']  # Left to right
        ))
        
        filtered = []
        seen_positions = []
        
        for match in matches:
            # Check if this match overlaps with any already filtered match
            is_duplicate = False
            for existing in filtered:
                if is_same_element(match, existing):
                    is_duplicate = True
                    logger.debug(f"Filtered duplicate: '{match['Label']}' at ({match['X']}, {match['Y']}) overlaps with '{existing['Label']}' at ({existing['X']}, {existing['Y']})")
                    break
            
            if not is_duplicate:
                # Check if this position has been seen before (with some tolerance)
                position = (match['X'], match['Y'])
                is_similar_position = any(
                    abs(pos[0] - position[0]) < 10 and abs(pos[1] - position[1]) < 10
                    for pos in seen_positions
                )
                
                if not is_similar_position:
                    filtered.append(match)
                    seen_positions.append(position)
                    logger.debug(f"Kept match: '{match['Label']}' at ({match['X']}, {match['Y']}) with confidence {match.get('Confidence', 0)}")
                else:
                    logger.debug(f"Filtered similar position: '{match['Label']}' at ({match['X']}, {match['Y']})")
        
        logger.info(f"Filtered {len(matches) - len(filtered)} duplicates from {len(matches)} matches")
        return filtered

    def select_best_match(matches: List[Dict], last_interacted_bbox: Optional[Dict] = None) -> Optional[Dict]:
        """Select the best match from a list of matches based on context and position."""
        if not matches:
            return None
            
        # If we have a last interacted bbox, prioritize matches near it
        if last_interacted_bbox:
            # Calculate distances to last interacted element
            distances = [(m, calculate_distance(m, last_interacted_bbox)) for m in matches]
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            best_match = distances[0][0]
            logger.info(f"Selected closest match to last interaction: '{best_match['Label']}' at distance {distances[0][1]:.2f}")
            return best_match
            
        # For login button specifically, prioritize the one below credentials
        if target_label.lower() == 'login':
            # Find the match with the highest Y coordinate (lowest on the page)
            matches.sort(key=lambda m: m['Y'], reverse=True)
            best_match = matches[0]
            logger.info(f"Selected login button at bottom: '{best_match['Label']}' at Y={best_match['Y']}")
            return best_match
            
        # Default: sort by confidence, size, and position
        matches.sort(key=lambda m: (
            -m.get('Confidence', 0),  # Higher confidence first
            -(m['Width'] * m['Height']),  # Larger elements first
            m['Y'],  # Higher up on the page first
            m['X']  # Left to right
        ))
        best_match = matches[0]
        logger.info(f"Selected best match based on confidence and position: '{best_match['Label']}'")
        return best_match

    # Collect all exact normalized matches for all variations
    exact_matches = []
    for result in ocr_results:
        norm_label = normalize_text(result['Label'])
        if norm_label in norm_variations:
            exact_matches.append(result)
            logger.debug(f"Found exact match for variation: '{result['Label']}'")
    
    if exact_matches:
        logger.info(f"Found {len(exact_matches)} exact normalized matches for '{target_label}' variations: {[m['Label'] for m in exact_matches]}")
        
        # Filter out duplicates
        exact_matches = filter_duplicates(exact_matches)
        logger.info(f"After filtering duplicates: {len(exact_matches)} matches remaining")
        
        if exact_matches:
            # Select best match based on context and position
            best_match = select_best_match(exact_matches, last_interacted_bbox)
            if best_match:
                return best_match

    # If no exact normalized match, try high similarity fuzzy match
    high_threshold = 0.70
    fuzzy_matches = []
    for result in ocr_results:
        norm_label = normalize_text(result['Label'])
        if not norm_label:
            continue
        # Check against all variations
        for norm_variation in norm_variations:
            similarity = difflib.SequenceMatcher(None, norm_variation, norm_label).ratio()
            if similarity >= high_threshold:
                fuzzy_matches.append((result, similarity))
                logger.debug(f"Fuzzy match candidate for '{target_label}': '{result['Label']}' with similarity {similarity:.2f}")

    if fuzzy_matches:
        logger.info(f"Found {len(fuzzy_matches)} fuzzy matches for '{target_label}' with threshold {high_threshold}: {[m[0]['Label'] for m in fuzzy_matches]}")
        # Sort by similarity descending
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        # Extract only results
        fuzzy_results = [m[0] for m in fuzzy_matches]
        
        # Filter out duplicates
        fuzzy_results = filter_duplicates(fuzzy_results)
        logger.info(f"After filtering duplicates: {len(fuzzy_results)} fuzzy matches remaining")
        
        if fuzzy_results:
            # Select best match based on context and position
            best_match = select_best_match(fuzzy_results, last_interacted_bbox)
            if best_match:
                return best_match

    # Log all OCR results for debugging
    logger.warning(f"No suitable match found for '{target_label}'. All OCR results:")
    for result in ocr_results:
        logger.warning(f"OCR result: '{result['Label']}' at ({result['X']}, {result['Y']}) with confidence {result.get('Confidence', 0)}")
    
    return None

def select_best_match(candidates: List[Dict], last_interacted_bbox: Optional[Dict]) -> Dict:
    """Select best match from candidates by highest confidence and proximity to last_interacted_bbox if provided."""
    if not candidates:
        return None
    # Sort by confidence descending
    candidates_sorted = sorted(candidates, key=lambda x: x.get('Confidence', 0), reverse=True)
    if last_interacted_bbox is None:
        return candidates_sorted[0]
    # Calculate distance to last_interacted_bbox center
    def distance(bbox1, bbox2):
        cx1 = bbox1['X'] + bbox1['Width'] / 2
        cy1 = bbox1['Y'] + bbox1['Height'] / 2
        cx2 = bbox2['X'] + bbox2['Width'] / 2
        cy2 = bbox2['Y'] + bbox2['Height'] / 2
        return math.hypot(cx1 - cx2, cy1 - cy2)
    # Find candidate with minimal distance among top confidence matches (top 3)
    top_candidates = candidates_sorted[:3]
    best_candidate = min(top_candidates, key=lambda c: distance(c, last_interacted_bbox))
    return best_candidate

def find_elements_by_label(config, target_label: str, ocr_results: List[Dict]) -> List[Dict]:
    """Find all matches for a label in OCR results using similarity."""
    if not ocr_results:
        return []
    norm_target = normalize_text(target_label)
    matches = []
    match_threshold = config.getfloat('Matching', 'label_match_threshold', fallback=0.9)

    for result in ocr_results:
        norm_current = normalize_text(result['Label'])
        if not norm_current:
            continue  # Skip empty lines

        similarity = difflib.SequenceMatcher(None, norm_target, norm_current).ratio()

        if similarity >= match_threshold:
            matches.append(result)
            logger.debug(f"Found a match for '{target_label}': '{result['Label']}' score={similarity:.2f}")

    if matches:
        logger.info(f"Found {len(matches)} matches for '{target_label}' (Threshold: {match_threshold:.2f})")
        return matches
    else:
        logger.warning(f"No suitable matches found for '{target_label}' (Threshold: {match_threshold:.2f})")
        return []

def calculate_click_coordinates(bbox: dict, element_type: str = 'button') -> Tuple[float, float]:
    """Calculate center click point. Maybe adjust logic for 'field' later."""
    center_x = bbox['X'] + bbox['Width'] / 2
    center_y = bbox['Y'] + bbox['Height'] / 2
    return center_x, center_y

# --- Coordinate-Based Actions ---

def click_coordinates(page: Page, bbox: dict, context=None):
    if not all(k in bbox for k in ['X', 'Y', 'Width', 'Height']): raise ValueError("Invalid bbox for click.")
    center_x, center_y = calculate_click_coordinates(bbox, 'button') # Assume button-like center click
    try: logger.info(f"Clicking element via coordinates: x={center_x:.0f}, y={center_y:.0f}"); page.mouse.click(center_x, center_y, delay=50); page.wait_for_timeout(500)
    except Exception as e: logger.error(f"Failed to click coordinates ({center_x:.0f}, {center_y:.0f}): {e}"); raise

def fill_coordinates(page: Page, bbox: Dict, value: str, context=None, clear_first=True):
    if not all(k in bbox for k in ['X', 'Y', 'Width', 'Height']): raise ValueError("Invalid bbox for fill.")
    center_x, center_y = calculate_click_coordinates(bbox, 'field') # Assume field-like center click for focus
    try:
        logger.info(f"Filling element via coordinates: x={center_x:.0f}, y={center_y:.0f}")
        page.mouse.click(center_x, center_y, delay=50); page.wait_for_timeout(100)
        if clear_first:
            page.keyboard.press("Control+A")
            page.keyboard.press("Delete")
            page.wait_for_timeout(50)
        page.keyboard.type(str(value), delay=50); page.wait_for_timeout(500)
    except Exception as e: logger.error(f"Failed to fill coordinates ({center_x:.0f}, {center_y:.0f}): {e}"); raise

def drag_and_drop_coordinates(page: Page, source_bbox: dict, target_bbox: dict, context=None):
    if not all(k in source_bbox for k in ['X', 'Y', 'Width', 'Height']) or not all(k in target_bbox for k in ['X', 'Y', 'Width', 'Height']):
        raise ValueError("Invalid bbox for drag and drop.")
    source_x, source_y = calculate_click_coordinates(source_bbox)
    target_x, target_y = calculate_click_coordinates(target_bbox)
    try:
        logger.info(f"Dragging from ({source_x:.0f}, {source_y:.0f}) to ({target_x:.0f}, {target_y:.0f})")
        page.mouse.move(source_x, source_y)
        page.mouse.down()
        page.mouse.move(target_x, target_y)
        page.mouse.up()
        page.wait_for_timeout(500)
    except Exception as e:
        logger.error(f"Failed to drag and drop from ({source_x:.0f}, {source_y:.0f}) to ({target_x:.0f}, {target_y:.0f}): {e}")
        raise

def slide_to_position(page: Page, slider_bbox: dict, target_x: int, context=None):
    """
    Slides a slider element horizontally to the target x coordinate.
    :param page: Playwright page object
    :param slider_bbox: Bounding box dict with keys X, Y, Width, Height
    :param target_x: Target x coordinate to slide to (absolute coordinate)
    :param context: Optional context for logging or screenshots
    """
    if not all(k in slider_bbox for k in ['X', 'Y', 'Width', 'Height']):
        raise ValueError("Invalid slider bounding box for sliding.")
    try:
        start_x = slider_bbox['X'] + slider_bbox['Width'] // 2
        start_y = slider_bbox['Y'] + slider_bbox['Height'] // 2
        logger.info(f"Sliding slider from ({start_x}, {start_y}) to target x={target_x}")
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(target_x, start_y, steps=10)
        page.mouse.up()
        page.wait_for_timeout(500)
        logger.info("Slider moved successfully.")
    except Exception as e:
        logger.error(f"Failed to slide slider to position {target_x}: {e}")
        raise

def get_coordinates_from_csv(config: configparser.ConfigParser, label_text: str) -> Optional[Dict]:
    """Retrieve coordinates for a specific label from the CSV file.
    
    Args:
        config: Configuration object containing CSV path
        label_text: The label text to search for
        
    Returns:
        Dictionary containing coordinates if found, None otherwise
    """
    try:
        # First try to get the path from the Paths section
        try:
            csv_path = config.get('Paths', 'coordinates_csv')
        except (configparser.NoSectionError, configparser.NoOptionError):
            # Fallback to General section if Paths section doesn't exist
            csv_path = config.get('General', 'runtime_ocr_csv')
            
        if not csv_path:
            logger.warning("No coordinates CSV path found in configuration")
            return None
            
        # Convert to absolute path if needed
        csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
            
        if not csv_path.exists():
            logger.warning(f"Coordinates CSV file not found at {csv_path}")
            return None
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Label'].lower() == label_text.lower():
                    return {
                        'X': int(row['X']),
                        'Y': int(row['Y']),
                        'Width': int(row['Width']),
                        'Height': int(row['Height']),
                        'Label': row['Label']
                    }
        logger.warning(f"No coordinates found for label '{label_text}'")
        return None
    except Exception as e:
        logger.error(f"Error reading coordinates from CSV: {str(e)}")
        return None

def save_screenshot(page: Page, config: configparser.ConfigParser, name: str) -> Optional[Path]:
    """
    Save a screenshot with the given name, replacing any existing file with the same name.
    
    Args:
        page: Playwright page object
        config: Configuration object
        name: Base name for the screenshot
        
    Returns:
        Path: Path to the saved screenshot, or None if failed
    """
    try:
        # Get screenshots directory from config or use default
        screenshots_dir = Path(config.get('Paths', 'screenshots_dir', fallback='screenshots'))
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old screenshots with the same base name
        base_name = name.split('_')[0]  # Get the base name without timestamp
        for old_file in screenshots_dir.glob(f"{base_name}_*.png"):
            try:
                old_file.unlink()
                logger.info(f"Removed old screenshot: {old_file}")
            except Exception as e:
                logger.warning(f"Failed to remove old screenshot {old_file}: {e}")
        
        # Generate new filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = screenshots_dir / filename
        
        # Take full page screenshot
        page.screenshot(path=str(filepath), full_page=True)
        logger.info(f"New screenshot saved: {filepath}")
        
        return filepath
    except Exception as e:
        logger.error(f"Failed to save screenshot: {e}")
        return None

def get_latest_screenshot(config: configparser.ConfigParser, base_name: str) -> Optional[str]:
    """
    Get the path to the latest screenshot with the given base name.
    
    Args:
        config: Configuration object
        base_name: Base name of the screenshot
        
    Returns:
        str: Path to the latest screenshot, or None if not found
    """
    try:
        screenshots_dir = config.get('Paths', 'screenshots_dir', fallback='screenshots')
        if not os.path.exists(screenshots_dir):
            return None
            
        matching_files = [
            f for f in os.listdir(screenshots_dir)
            if f.startswith(f"{base_name}_") and f.endswith('.png')
        ]
        
        if not matching_files:
            return None
            
        # Sort by timestamp (newest first)
        latest_file = sorted(matching_files, reverse=True)[0]
        return os.path.join(screenshots_dir, latest_file)
    except Exception as e:
        logger.error(f"Failed to get latest screenshot: {e}")
        return None
