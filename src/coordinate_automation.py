import csv
import math
from typing import Dict, List, Tuple
from playwright.sync_api import Page, expect
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import cv2
import os
from datetime import datetime

class CoordinateAutomation:
    def __init__(self, page: Page, csv_path: str = "detected_coordinates.csv"):
        self.page = page
        self.csv_path = csv_path
        self.coordinates: Dict[str, List[Tuple[int, int]]] = {}
        self.last_action_coords = None
        self.reader = easyocr.Reader(['en'])
        self.load_coordinates()

    def load_coordinates(self):
        """Load coordinates from CSV file"""
        if not os.path.exists(self.csv_path):
            return
        
        with open(self.csv_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 3:
                    label = row[0]
                    x, y = int(row[1]), int(row[2])
                    if label not in self.coordinates:
                        self.coordinates[label] = []
                    self.coordinates[label].append((x, y))

    def save_coordinates(self):
        """Save coordinates to CSV file"""
        with open(self.csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for label, coords_list in self.coordinates.items():
                for x, y in coords_list:
                    writer.writerow([label, x, y])

    def calculate_distance(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two coordinates"""
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

    def find_nearest_coordinate(self, label: str) -> Tuple[int, int]:
        """Find the coordinate nearest to the last action"""
        if label not in self.coordinates:
            raise ValueError(f"Label '{label}' not found in coordinates")
        
        if not self.last_action_coords:
            return self.coordinates[label][0]
        
        return min(self.coordinates[label], 
                  key=lambda coord: self.calculate_distance(coord, self.last_action_coords))

    def perform_ocr(self):
        """Perform OCR on the current page and update coordinates"""
        # Take screenshot
        screenshot_path = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.page.screenshot(path=screenshot_path)
        
        # Read image
        image = cv2.imread(screenshot_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Perform OCR
        results = self.reader.readtext(gray)
        
        # Update coordinates
        self.coordinates.clear()
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Confidence threshold
                x = int((bbox[0][0] + bbox[2][0]) / 2)
                y = int((bbox[0][1] + bbox[2][1]) / 2)
                if text not in self.coordinates:
                    self.coordinates[text] = []
                self.coordinates[text].append((x, y))
        
        self.save_coordinates()

    def click(self, label: str):
        """Click at the coordinate of the given label"""
        coord = self.find_nearest_coordinate(label)
        self.page.mouse.click(coord[0], coord[1])
        self.last_action_coords = coord

    def type(self, label: str, text: str):
        """Type text at the coordinate of the given label"""
        coord = self.find_nearest_coordinate(label)
        self.page.mouse.click(coord[0], coord[1])
        self.page.keyboard.type(text)
        self.last_action_coords = coord

    def check_page_change(self):
        """Check if page has changed and perform OCR if needed"""
        # Implement your page change detection logic here
        # For example, you could compare screenshots or check for specific elements
        # For now, we'll just perform OCR after each action
        self.perform_ocr()

    def navigate(self, url: str):
        """Navigate to a URL and perform OCR"""
        self.page.goto(url)
        self.perform_ocr()
        self.last_action_coords = None 