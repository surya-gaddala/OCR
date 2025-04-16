# OCR Form Automation Project

This project automates form interactions using OCR (Optical Character Recognition) to detect field labels and buttons. It uses Playwright for browser automation and Tesseract/EasyOCR for text detection.

## Features
- Coordinate-based automation using OCR
- Automatic OCR updates during navigation
- Smart handling of duplicate labels
- Comprehensive logging and error handling
- Retry mechanisms for reliability
- Screenshot capture for debugging

## Prerequisites
1. Python 3.10 or higher
2. Tesseract OCR:
   - Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to `C:\Program Files\Tesseract-OCR`
   - Add to system PATH
3. Git (optional, for version control)

## Setup Instructions

1. Clone the repository (if using Git):
   ```bash
   git clone <repository-url>
   cd ocr_automation
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Install Playwright browsers:
   ```bash
   playwright install
   ```

5. Clean Python cache:
   ```bash
   # On Windows:
   del /S *.pyc
   # On Unix/MacOS:
   find . -type f -name "*.pyc" -delete
   ```

6. Configure the project:
   - Update `config/config.ini` with your settings
   - Set Tesseract path if not in system PATH
   - Adjust confidence thresholds
   - Configure OCR engine preferences

## Running Tests

1. Run all tests:
   ```bash
   behave
   ```

2. Run specific feature:
   ```bash
   behave features/login.feature
   ```

3. Run with specific tags:
   ```bash
   behave --tags=@login
   ```

## Project Structure
```
ocr_automation/
├── config/
│   └── config.ini
├── features/
│   ├── environment.py
│   ├── steps/
│   │   └── login_steps.py
│   └── login.feature
├── logs/
│   └── automation.log
├── screenshots/
├── src/
│   └── automation_ocr_utils.py
├── requirements.txt
└── README.md
```

## Configuration
The `config.ini` file controls various aspects of the automation:
- OCR settings (engine, confidence thresholds)
- Browser settings (timeouts, retries)
- Logging configuration
- Screenshot settings

## Troubleshooting

1. OCR Issues:
   - Verify Tesseract installation
   - Check system PATH
   - Adjust confidence thresholds in config
   - Check image preprocessing settings

2. Browser Automation Issues:
   - Verify Playwright installation
   - Check browser compatibility
   - Update browser drivers if needed
   - Check network connectivity

3. General Issues:
   - Check logs in `logs/automation.log`
   - Review screenshots in `screenshots/`
   - Verify coordinate CSV file

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.