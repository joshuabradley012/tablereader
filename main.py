try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

print(pytesseract.image_to_string(Image.open('test.png')))
print(pytesseract.image_to_boxes(Image.open('test.png')))
print(pytesseract.image_to_data(Image.open('test.png')))
