"""
QR Code Generator for Chrome Flags URL
Helps users quickly access the Chrome flags page to enable insecure origins.
"""

import qrcode
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import socket

def get_local_ip():
    """Get the local IP address"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "localhost"

def create_chrome_flags_qr():
    """Create QR code for Chrome flags URL"""
    # Chrome flags URL
    chrome_flags_url = "chrome://flags/#unsafely-treat-insecure-origin-as-secure"
    
    # Create QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(chrome_flags_url)
    qr.make(fit=True)
    
    # Create QR code image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to RGB if needed
    if qr_img.mode != 'RGB':
        qr_img = qr_img.convert('RGB')
    
    # Create a larger image with instructions
    width, height = qr_img.size
    new_height = height + 100
    final_img = Image.new('RGB', (width, new_height), 'white')
    
    # Paste QR code
    final_img.paste(qr_img, (0, 0, width, height))
    
    # Add text instructions
    draw = ImageDraw.Draw(final_img)
    
    try:
        # Try to use a nice font
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add instructions
    text1 = "1. Scan this QR code"
    text2 = "2. Enable the flag"
    text3 = "3. Add your laptop URL"
    text4 = f"4. URL: http://{get_local_ip()}:8502"
    
    y_offset = height + 10
    draw.text((10, y_offset), text1, fill="black", font=font)
    draw.text((10, y_offset + 20), text2, fill="black", font=font)
    draw.text((10, y_offset + 40), text3, fill="black", font=font)
    draw.text((10, y_offset + 60), text4, fill="blue", font=font_small)
    
    return final_img

def create_app_url_qr():
    """Create QR code for the app URL"""
    local_ip = get_local_ip()
    app_url = f"http://{local_ip}:8502"
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(app_url)
    qr.make(fit=True)
    
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to RGB if needed
    if qr_img.mode != 'RGB':
        qr_img = qr_img.convert('RGB')
    
    # Add title
    width, height = qr_img.size
    new_height = height + 60
    final_img = Image.new('RGB', (width, new_height), 'white')
    final_img.paste(qr_img, (0, 40, width, height + 40))
    
    draw = ImageDraw.Draw(final_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Center the title
    title = "AdMyVision App"
    try:
        title_width = draw.textlength(title, font=font)
    except:
        # Fallback for older PIL versions
        title_width = draw.textsize(title, font=font)[0]
    
    x = (width - title_width) // 2
    draw.text((x, 10), title, fill="black", font=font)
    
    return final_img

if __name__ == "__main__":
    print("Creating QR codes...")
    
    # Create Chrome flags QR code
    chrome_qr = create_chrome_flags_qr()
    chrome_qr.save("chrome_flags_qr.png")
    print("✅ Chrome flags QR code saved as 'chrome_flags_qr.png'")
    
    # Create app URL QR code
    app_qr = create_app_url_qr()
    app_qr.save("app_url_qr.png")
    print("✅ App URL QR code saved as 'app_url_qr.png'")
    
    print(f"\n📱 Instructions for phone users:")
    print(f"1. Scan 'chrome_flags_qr.png' to open Chrome flags")
    print(f"2. Enable the 'unsafely-treat-insecure-origin-as-secure' flag")
    print(f"3. Add: http://{get_local_ip()}:8502")
    print(f"4. Restart Chrome")
    print(f"5. Scan 'app_url_qr.png' to open AdMyVision")
    
    # Show the images if possible
    try:
        chrome_qr.show()
        app_qr.show()
    except:
        print("Could not display images automatically")
