# 🚀 LexiBot Computer Vision - Quick Start

## What This Project Does

This system detects famous artworks in real-time using your webcam! It can identify:
- 🎨 **Mona Lisa** by Leonardo da Vinci
- 🌙 **The Starry Night** by Vincent van Gogh
- 🗽 **Liberty Leading the People** by Eugène Delacroix
- 😱 **The Scream** by Edvard Munch
- 🌻 **Sunflowers** by Vincent van Gogh

## 📱 Demo Instructions

1. **Run the main app**: `python scripts/real_time_detection.py`
2. **Show artwork images** to your camera:
   - Use your phone to display artwork images from Google
   - Print out artwork images 
   - Use artwork books or posters
   - Try computer screen images
3. **Watch the magic happen**: Green boxes will appear around detected artworks with:
   - 🏷️ Artwork name
   - 📊 Confidence percentage
   - 📏 Estimated distance

## ⚡ Quick Tests

**Test your camera first:**
```bash
python test_camera.py
```

**Test art detection without camera:**
```bash
python test_custom_art_detection.py
```

**Demo MQTT communication:**
```bash
python demo_mqtt_art.py
```

## 🎯 Tips for Best Results

- ✅ **Good lighting** helps detection accuracy
- ✅ **Clear, unblurred images** work best
- ✅ **Full artwork visible** gives better results
- ✅ **60-200cm distance** from camera is optimal
- ❌ Avoid very dark or very bright environments
- ❌ Don't cover parts of the artwork

## 🆘 Need Help?

- Press **ESC** to exit any application
- Check the main **README.md** for detailed instructions
- Look at the **troubleshooting section** for common issues

---

**Happy art detecting! 🎨✨**
