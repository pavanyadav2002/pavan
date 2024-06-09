from PIL import Image
from pytesseract import pytesseract
import enum


class OS(enum.Enum):
    mac = 0
    windows = 1


class Language(enum.Enum):
    ENG = 'eng'
    RUS = 'rus'
    ITA = 'ita'


class ImageReader:

    def __init__(self,os: OS):
        if os == OS.mac:
            #tesseract is already installed via homebrew
            print('Running on: MAC\n')

            if os == OS.windows:
                windows_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                pytesseract.tesseract_cmd = windows_path
                print('Running on: WINDOWS\n')

    def extract_text(self, image: str, lang: str):
        img = Image.open(image)
        extracted_text = pytesseract.image_to_string(img, lang=lang)
        return extracted_text    


if __name__ == '__main__':
    ir = ImageReader(OS.mac)
    text = ir.extract_text('/Users/pavanyadav/Desktop/infosysvenv/adharcard.jpeg', lang='eng')
    print(text)