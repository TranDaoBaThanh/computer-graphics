import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from PIL import Image, ImageTk
import numpy as np
import sys
import os
import colorsys

# Kích thước cửa sổ OpenGL
WIDTH, HEIGHT = 800, 600

# Các chế độ hiển thị
MODE_ORIGINAL = 0
MODE_GRAYSCALE = 1
MODE_HSV = 2
MODE_CMYK = 3
MODE_RGB_ADJUST = 4
MODE_LAB = 5
MODE_YUV = 6
MODE_HSL = 7
MODE_YCBCR = 8
MODE_SRGB = 9
MODE_ADOBE_RGB = 10
MODE_XYZ = 11
MODE_NEGATIVE = 12
MODE_SEPIA = 13
MODE_CUSTOM = 14

# Các biến điều chỉnh toàn cục
current_mode = MODE_ORIGINAL
original_image_path = None
custom_filter_values = {
    "red": 1.0, "green": 1.0, "blue": 1.0,    # RGB
    "hue": 0.0, "saturation": 1.0, "value": 1.0,   # HSV
    "lightness": 1.0,                          # HSL
    "cyan": 0.0, "magenta": 0.0, "yellow": 0.0, "key": 0.0,  # CMYK
    "y": 1.0, "u": 0.0, "v": 0.0,             # YUV
    "cb": 0.0, "cr": 0.0,                     # YCbCr
    "l": 1.0, "a": 0.0, "b": 0.0,             # Lab
    "contrast": 1.0, "brightness": 0.0, "gamma": 1.0,  # Các điều chỉnh chung
    "sepia_intensity": 1.0,                   # Sepia
    "x": 1.0, "y_xyz": 1.0, "z": 1.0,          # XYZ
}

window = None
shader = None
texture = None
VAO = None
should_close = False

# Vertex shader
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoord;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(position, 1.0);
    TexCoord = texCoord;
}
"""

# Fragment shader với chuyển đổi màu nâng cao
fragment_shader_source = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ourTexture;
uniform int mode;
uniform float redAdjust;
uniform float greenAdjust;
uniform float blueAdjust;
uniform float hueAdjust;
uniform float saturationAdjust;
uniform float valueAdjust;
uniform float lightnessAdjust;
uniform float cyanAdjust;
uniform float magentaAdjust;
uniform float yellowAdjust;
uniform float keyAdjust;
uniform float yAdjust;
uniform float uAdjust;
uniform float vAdjust;
uniform float cbAdjust;
uniform float crAdjust;
uniform float lAdjust;
uniform float aAdjust;
uniform float bAdjust;
uniform float contrastAdjust;
uniform float brightnessAdjust;
uniform float gammaAdjust;
uniform float sepiaIntensity;
uniform float xAdjust;
uniform float yAdjust_xyz;
uniform float zAdjust;

// Hàm chuyển từ RGB sang HSV
vec3 rgb2hsv(vec3 c)
{
    float cMax = max(c.r, max(c.g, c.b));
    float cMin = min(c.r, min(c.g, c.b));
    float delta = cMax - cMin;
    float h = 0.0;
    if(delta < 0.00001)
        h = 0.0;
    else if(cMax == c.r)
        h = mod((60.0 * ((c.g - c.b) / delta) + 360.0), 360.0);
    else if(cMax == c.g)
        h = mod((60.0 * ((c.b - c.r) / delta) + 120.0), 360.0);
    else if(cMax == c.b)
        h = mod((60.0 * ((c.r - c.g) / delta) + 240.0), 360.0);
    
    float s = (cMax < 0.00001 ? 0.0 : delta / cMax);
    float v = cMax;
    return vec3(h, s, v);
}

// Hàm chuyển từ HSV sang RGB
vec3 hsv2rgb(vec3 c)
{
    float h = c.x;
    float s = c.y;
    float v = c.z;
    float c_val = v * s;
    float x = c_val * (1.0 - abs(mod(h / 60.0, 2.0) - 1.0));
    float m = v - c_val;
    vec3 rgb;
    if(h < 60.0)
        rgb = vec3(c_val, x, 0.0);
    else if(h < 120.0)
        rgb = vec3(x, c_val, 0.0);
    else if(h < 180.0)
        rgb = vec3(0.0, c_val, x);
    else if(h < 240.0)
        rgb = vec3(0.0, x, c_val);
    else if(h < 300.0)
        rgb = vec3(x, 0.0, c_val);
    else
        rgb = vec3(c_val, 0.0, x);
    return rgb + vec3(m);
}

// Hàm chuyển từ RGB sang HSL
vec3 rgb2hsl(vec3 color)
{
    float maxColor = max(max(color.r, color.g), color.b);
    float minColor = min(min(color.r, color.g), color.b);
    float delta = maxColor - minColor;
    
    float h = 0.0;
    float s = 0.0;
    float l = (maxColor + minColor) / 2.0;
    
    if (delta > 0.0)
    {
        s = (l < 0.5) ? (delta / (maxColor + minColor)) : (delta / (2.0 - maxColor - minColor));
        
        if (maxColor == color.r)
            h = (color.g - color.b) / delta + (color.g < color.b ? 6.0 : 0.0);
        else if (maxColor == color.g)
            h = (color.b - color.r) / delta + 2.0;
        else
            h = (color.r - color.g) / delta + 4.0;
            
        h /= 6.0;
    }
    
    return vec3(h * 360.0, s, l);
}

// Hàm chuyển từ HSL sang RGB
float hue2rgb(float p, float q, float t)
{
    if (t < 0.0) t += 1.0;
    if (t > 1.0) t -= 1.0;
    if (t < 1.0/6.0) return p + (q - p) * 6.0 * t;
    if (t < 1.0/2.0) return q;
    if (t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
    return p;
}

vec3 hsl2rgb(vec3 hsl)
{
    float h = hsl.x / 360.0;
    float s = hsl.y;
    float l = hsl.z;
    
    if (s == 0.0)
        return vec3(l);
    
    float q = (l < 0.5) ? (l * (1.0 + s)) : (l + s - l * s);
    float p = 2.0 * l - q;
    
    return vec3(
        hue2rgb(p, q, h + 1.0/3.0),
        hue2rgb(p, q, h),
        hue2rgb(p, q, h - 1.0/3.0)
    );
}

// Hàm chuyển từ RGB sang CMYK
vec4 rgb2cmyk(vec3 rgb)
{
    float k = 1.0 - max(max(rgb.r, rgb.g), rgb.b);
    float c = (1.0 - rgb.r - k) / (1.0 - k);
    float m = (1.0 - rgb.g - k) / (1.0 - k);
    float y = (1.0 - rgb.b - k) / (1.0 - k);
    
    if (k == 1.0) {
        c = 0.0;
        m = 0.0;
        y = 0.0;
    }
    
    return vec4(c, m, y, k);
}

// Hàm chuyển từ CMYK sang RGB
vec3 cmyk2rgb(vec4 cmyk)
{
    float c = cmyk.x;
    float m = cmyk.y;
    float y = cmyk.z;
    float k = cmyk.w;
    
    float r = (1.0 - c) * (1.0 - k);
    float g = (1.0 - m) * (1.0 - k);
    float b = (1.0 - y) * (1.0 - k);
    
    return vec3(r, g, b);
}

// Hàm chuyển từ RGB sang YUV
vec3 rgb2yuv(vec3 rgb)
{
    float y =  0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    float u = -0.147 * rgb.r - 0.289 * rgb.g + 0.436 * rgb.b;
    float v =  0.615 * rgb.r - 0.515 * rgb.g - 0.100 * rgb.b;
    
    return vec3(y, u, v);
}

// Hàm chuyển từ YUV sang RGB
vec3 yuv2rgb(vec3 yuv)
{
    float r = yuv.x + 1.140 * yuv.z;
    float g = yuv.x - 0.395 * yuv.y - 0.581 * yuv.z;
    float b = yuv.x + 2.032 * yuv.y;
    
    return vec3(r, g, b);
}

// Hàm chuyển đổi RGB sang YCbCr
vec3 rgb2ycbcr(vec3 rgb)
{
    float y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    float cb = 0.5 - 0.168736 * rgb.r - 0.331264 * rgb.g + 0.5 * rgb.b;
    float cr = 0.5 + 0.5 * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b;
    
    return vec3(y, cb, cr);
}

// Hàm chuyển đổi YCbCr sang RGB
vec3 ycbcr2rgb(vec3 ycbcr)
{
    float y = ycbcr.x;
    float cb = ycbcr.y - 0.5;
    float cr = ycbcr.z - 0.5;
    
    float r = y + 1.402 * cr;
    float g = y - 0.344136 * cb - 0.714136 * cr;
    float b = y + 1.772 * cb;
    
    return vec3(r, g, b);
}

// Hàm chuyển đổi RGB sang XYZ
vec3 rgb2xyz(vec3 rgb)
{
    // sRGB to XYZ matrix
    vec3 xyz;
    xyz.x = 0.4124 * rgb.r + 0.3576 * rgb.g + 0.1805 * rgb.b;
    xyz.y = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    xyz.z = 0.0193 * rgb.r + 0.1192 * rgb.g + 0.9505 * rgb.b;
    return xyz;
}

// Hàm chuyển đổi XYZ sang RGB
vec3 xyz2rgb(vec3 xyz)
{
    // XYZ to sRGB matrix
    vec3 rgb;
    rgb.r =  3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z;
    rgb.g = -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z;
    rgb.b =  0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z;
    return rgb;
}

// Hàm chuyển đổi XYZ sang Lab
vec3 xyz2lab(vec3 xyz)
{
    // D65 reference white
    float xn = 0.95047;
    float yn = 1.0;
    float zn = 1.08883;
    
    float x = xyz.x / xn;
    float y = xyz.y / yn;
    float z = xyz.z / zn;
    
    x = (x > 0.008856) ? pow(x, 1.0/3.0) : (7.787 * x + 16.0/116.0);
    y = (y > 0.008856) ? pow(y, 1.0/3.0) : (7.787 * y + 16.0/116.0);
    z = (z > 0.008856) ? pow(z, 1.0/3.0) : (7.787 * z + 16.0/116.0);
    
    float L = (116.0 * y) - 16.0;
    float a = 500.0 * (x - y);
    float b = 200.0 * (y - z);
    
    return vec3(L, a, b);
}

// Hàm chuyển đổi Lab sang XYZ
vec3 lab2xyz(vec3 lab)
{
    float y = (lab.x + 16.0) / 116.0;
    float x = lab.y / 500.0 + y;
    float z = y - lab.z / 200.0;
    
    float x3 = x * x * x;
    float y3 = y * y * y;
    float z3 = z * z * z;
    
    x = (x3 > 0.008856) ? x3 : (x - 16.0/116.0) / 7.787;
    y = (y3 > 0.008856) ? y3 : (y - 16.0/116.0) / 7.787;
    z = (z3 > 0.008856) ? z3 : (z - 16.0/116.0) / 7.787;
    
    // D65 reference white
    float xn = 0.95047;
    float yn = 1.0;
    float zn = 1.08883;
    
    return vec3(x * xn, y * yn, z * zn);
}

// Hàm chuyển RGB sang sRGB
vec3 rgb2srgb(vec3 rgb)
{
    vec3 srgb;
    for (int i = 0; i < 3; i++) {
        if (rgb[i] <= 0.0031308)
            srgb[i] = 12.92 * rgb[i];
        else
            srgb[i] = 1.055 * pow(rgb[i], 1.0/2.4) - 0.055;
    }
    return srgb;
}

// Hàm chuyển sRGB sang RGB
vec3 srgb2rgb(vec3 srgb)
{
    vec3 rgb;
    for (int i = 0; i < 3; i++) {
        if (srgb[i] <= 0.04045)
            rgb[i] = srgb[i] / 12.92;
        else
            rgb[i] = pow((srgb[i] + 0.055) / 1.055, 2.4);
    }
    return rgb;
}

// Hàm chuyển RGB sang Adobe RGB
vec3 rgb2adobergb(vec3 rgb)
{
    vec3 adobergb;
    for (int i = 0; i < 3; i++) {
        adobergb[i] = pow(rgb[i], 2.2);
    }
    return adobergb;
}

// Hàm chuyển Adobe RGB sang RGB
vec3 adobergb2rgb(vec3 adobergb)
{
    vec3 rgb;
    for (int i = 0; i < 3; i++) {
        rgb[i] = pow(adobergb[i], 1.0/2.2);
    }
    return rgb;
}

// Hàm hiệu chỉnh độ tương phản
vec3 adjustContrast(vec3 color, float contrast)
{
    return (color - 0.5) * contrast + 0.5;
}

// Hàm hiệu chỉnh độ sáng
vec3 adjustBrightness(vec3 color, float brightness)
{
    return color + brightness;
}

// Hàm hiệu chỉnh gamma
vec3 adjustGamma(vec3 color, float gamma)
{
    return pow(color, vec3(1.0 / gamma));
}

// Hàm chuyển đổi sang hiệu ứng sepia
vec3 sepia(vec3 color, float intensity)
{
    vec3 sepia = vec3(
        dot(color, vec3(0.393, 0.769, 0.189)),
        dot(color, vec3(0.349, 0.686, 0.168)),
        dot(color, vec3(0.272, 0.534, 0.131))
    );
    return mix(color, sepia, intensity);
}

// Đảo ngược màu (negative)
vec3 negative(vec3 color)
{
    return 1.0 - color;
}

void main()
{
    vec3 originalColor = texture(ourTexture, TexCoord).rgb;
    vec3 finalColor = originalColor;
    
    if(mode == 1) // Grayscale
    {
        float gray = dot(originalColor, vec3(0.299, 0.587, 0.114));
        finalColor = vec3(gray);
    }
    else if(mode == 2) // HSV
    {
        vec3 hsv = rgb2hsv(originalColor);
        hsv.x = mod(hsv.x + hueAdjust, 360.0);
        hsv.y = clamp(hsv.y * saturationAdjust, 0.0, 1.0);
        hsv.z = clamp(hsv.z * valueAdjust, 0.0, 1.0);
        finalColor = hsv2rgb(hsv);
    }
    else if(mode == 3) // CMYK
    {
        vec4 cmyk = rgb2cmyk(originalColor);
        cmyk.x = clamp(cmyk.x + cyanAdjust, 0.0, 1.0);
        cmyk.y = clamp(cmyk.y + magentaAdjust, 0.0, 1.0);
        cmyk.z = clamp(cmyk.z + yellowAdjust, 0.0, 1.0);
        cmyk.w = clamp(cmyk.w + keyAdjust, 0.0, 1.0);
        finalColor = cmyk2rgb(cmyk);
    }
    else if(mode == 4) // RGB điều chỉnh
    {
        finalColor = vec3(
            originalColor.r * redAdjust,
            originalColor.g * greenAdjust,
            originalColor.b * blueAdjust
        );
    }
    else if(mode == 5) // LAB
    {
        vec3 xyz = rgb2xyz(originalColor);
        vec3 lab = xyz2lab(xyz);
        lab.x = clamp(lab.x * lAdjust, 0.0, 100.0);
        lab.y = clamp(lab.y + aAdjust * 128.0, -128.0, 127.0);
        lab.z = clamp(lab.z + bAdjust * 128.0, -128.0, 127.0);
        xyz = lab2xyz(lab);
        finalColor = xyz2rgb(xyz);
    }
    else if(mode == 6) // YUV
    {
        vec3 yuv = rgb2yuv(originalColor);
        yuv.x = clamp(yuv.x * yAdjust, 0.0, 1.0);
        yuv.y = clamp(yuv.y + uAdjust, -0.5, 0.5);
        yuv.z = clamp(yuv.z + vAdjust, -0.5, 0.5);
        finalColor = yuv2rgb(yuv);
    }
    else if(mode == 7) // HSL
    {
        vec3 hsl = rgb2hsl(originalColor);
        hsl.x = mod(hsl.x + hueAdjust, 360.0);
        hsl.y = clamp(hsl.y * saturationAdjust, 0.0, 1.0);
        hsl.z = clamp(hsl.z * lightnessAdjust, 0.0, 1.0);
        finalColor = hsl2rgb(hsl);
    }
    else if(mode == 8) // YCbCr
    {
        vec3 ycbcr = rgb2ycbcr(originalColor);
        ycbcr.x = clamp(ycbcr.x * yAdjust, 0.0, 1.0);
        ycbcr.y = clamp(ycbcr.y + cbAdjust, 0.0, 1.0);
        ycbcr.z = clamp(ycbcr.z + crAdjust, 0.0, 1.0);
        finalColor = ycbcr2rgb(ycbcr);
    }
    else if(mode == 9) // sRGB
    {
        vec3 srgb = rgb2srgb(originalColor);
        srgb = adjustContrast(srgb, contrastAdjust);
        srgb = adjustBrightness(srgb, brightnessAdjust);
        finalColor = srgb2rgb(srgb);
    }
    else if(mode == 10) // Adobe RGB
    {
        vec3 adobergb = rgb2adobergb(originalColor);
        adobergb = adjustContrast(adobergb, contrastAdjust);
        adobergb = adjustBrightness(adobergb, brightnessAdjust);
        finalColor = adobergb2rgb(adobergb);
    }
    else if(mode == 11) // XYZ
    {
        vec3 xyz = rgb2xyz(originalColor);
        xyz.x *= xAdjust;
        xyz.y *= yAdjust_xyz;
        xyz.z *= zAdjust;
        finalColor = xyz2rgb(xyz);
    }
    else if(mode == 12) // Negative
    {
        finalColor = negative(originalColor);
    }
    else if(mode == 13) // Sepia
    {
        finalColor = sepia(originalColor, sepiaIntensity);
    }
    else if(mode == 14) // Custom (kết hợp nhiều hiệu ứng)
    {
        finalColor = originalColor;

        // RGB adjustment
        finalColor = vec3(
            finalColor.r * redAdjust,
            finalColor.g * greenAdjust,
            finalColor.b * blueAdjust
        );
        
        // Contrast, brightness, gamma
        finalColor = adjustContrast(finalColor, contrastAdjust);
        finalColor = adjustBrightness(finalColor, brightnessAdjust);
        finalColor = adjustGamma(finalColor, gammaAdjust);
        
        // Tùy chọn: thêm hiệu ứng HSV nếu người dùng muốn
        if (saturationAdjust != 1.0 || hueAdjust != 0.0 || valueAdjust != 1.0) {
            vec3 hsv = rgb2hsv(finalColor);
            hsv.x = mod(hsv.x + hueAdjust, 360.0);
            hsv.y = clamp(hsv.y * saturationAdjust, 0.0, 1.0);
            hsv.z = clamp(hsv.z * valueAdjust, 0.0, 1.0);
            finalColor = hsv2rgb(hsv);
        }
    }
    
    // Luôn đảm bảo màu nằm trong khoảng hợp lệ [0,1]
    finalColor = clamp(finalColor, 0.0, 1.0);
    
    FragColor = vec4(finalColor, 1.0);
}
"""

def load_texture(path):
    try:
        image = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    except Exception as e:
        print("Không thể tải ảnh:", e)
        return None
    
    # Tạo thumbnail để hiển thị trong UI
    thumbnail = Image.open(path)
    thumbnail.thumbnail((200, 200))
    thumbnail_tk = ImageTk.PhotoImage(thumbnail)
    
    image = image.convert("RGB")
    img_data = np.array(image, dtype=np.uint8)
    width, height = image.size

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Thiết lập tham số texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # Tải dữ liệu texture lên GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture, thumbnail_tk, (width, height)

def setup_opengl(image_path):
    global window, shader, texture, VAO, should_close

    if not glfw.init():
        print("Không khởi tạo được GLFW")
        return False, None, None

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, False)

    window = glfw.create_window(WIDTH, HEIGHT, "OpenGL Renderer", None, None)
    if not window:
        print("Không tạo được cửa sổ GLFW")
        glfw.terminate()
        return False, None, None

    glfw.make_context_current(window)

    vertex_shader = OpenGL.GL.shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = OpenGL.GL.shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader = OpenGL.GL.shaders.compileProgram(vertex_shader, fragment_shader)

    vertices = np.array([
        1.0,  1.0, 0.0,    1.0, 1.0,
        1.0, -1.0, 0.0,    1.0, 0.0,
       -1.0, -1.0, 0.0,    0.0, 0.0,
       -1.0,  1.0, 0.0,    0.0, 1.0
    ], dtype=np.float32)

    indices = np.array([0, 1, 3, 1, 2, 3], dtype=np.uint32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    texture_data = load_texture(image_path)
    if texture_data is None:
        glfw.terminate()
        return False, None, None
    
    texture, thumbnail_tk, image_size = texture_data
    return True, thumbnail_tk, image_size

def render_frame():
    if window is None:
        return None
    
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shader)

    # Cập nhật các uniform
    glUniform1i(glGetUniformLocation(shader, "mode"), current_mode)
    
    # RGB Adjust
    glUniform1f(glGetUniformLocation(shader, "redAdjust"), custom_filter_values["red"])
    glUniform1f(glGetUniformLocation(shader, "greenAdjust"), custom_filter_values["green"])
    glUniform1f(glGetUniformLocation(shader, "blueAdjust"), custom_filter_values["blue"])
    
    # HSV Adjust
    glUniform1f(glGetUniformLocation(shader, "hueAdjust"), custom_filter_values["hue"])
    glUniform1f(glGetUniformLocation(shader, "saturationAdjust"), custom_filter_values["saturation"])
    glUniform1f(glGetUniformLocation(shader, "valueAdjust"), custom_filter_values["value"])
    
    # HSL Adjust
    glUniform1f(glGetUniformLocation(shader, "lightnessAdjust"), custom_filter_values["lightness"])
    
    # CMYK Adjust
    glUniform1f(glGetUniformLocation(shader, "cyanAdjust"), custom_filter_values["cyan"])
    glUniform1f(glGetUniformLocation(shader, "magentaAdjust"), custom_filter_values["magenta"])
    glUniform1f(glGetUniformLocation(shader, "yellowAdjust"), custom_filter_values["yellow"])
    glUniform1f(glGetUniformLocation(shader, "keyAdjust"), custom_filter_values["key"])
    
    # YUV Adjust
    glUniform1f(glGetUniformLocation(shader, "yAdjust"), custom_filter_values["y"])
    glUniform1f(glGetUniformLocation(shader, "uAdjust"), custom_filter_values["u"])
    glUniform1f(glGetUniformLocation(shader, "vAdjust"), custom_filter_values["v"])
    
    # YCbCr Adjust
    glUniform1f(glGetUniformLocation(shader, "cbAdjust"), custom_filter_values["cb"])
    glUniform1f(glGetUniformLocation(shader, "crAdjust"), custom_filter_values["cr"])
    
    # Lab Adjust
    glUniform1f(glGetUniformLocation(shader, "lAdjust"), custom_filter_values["l"])
    glUniform1f(glGetUniformLocation(shader, "aAdjust"), custom_filter_values["a"])
    glUniform1f(glGetUniformLocation(shader, "bAdjust"), custom_filter_values["b"])
    
    # Contrast, Brightness, Gamma
    glUniform1f(glGetUniformLocation(shader, "contrastAdjust"), custom_filter_values["contrast"])
    glUniform1f(glGetUniformLocation(shader, "brightnessAdjust"), custom_filter_values["brightness"])
    glUniform1f(glGetUniformLocation(shader, "gammaAdjust"), custom_filter_values["gamma"])
    
    # Sepia
    glUniform1f(glGetUniformLocation(shader, "sepiaIntensity"), custom_filter_values["sepia_intensity"])
    
    # XYZ
    glUniform1f(glGetUniformLocation(shader, "xAdjust"), custom_filter_values["x"])
    glUniform1f(glGetUniformLocation(shader, "yAdjust_xyz"), custom_filter_values["y_xyz"])
    glUniform1f(glGetUniformLocation(shader, "zAdjust"), custom_filter_values["z"])

    # Kích hoạt texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture)
    glUniform1i(glGetUniformLocation(shader, "ourTexture"), 0)

    # Vẽ hình chữ nhật
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    # Đọc dữ liệu hình ảnh từ framebuffer
    data = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (WIDTH, HEIGHT), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    return image

def cleanup_opengl():
    global window, shader, texture, VAO, should_close
    
    if window is not None:
        glfw.make_context_current(window)
        
        if VAO is not None:
            glDeleteVertexArrays(1, [VAO])
        
        if texture is not None:
            glDeleteTextures(1, [texture])
        
        if shader is not None:
            glDeleteProgram(shader)
        
        glfw.destroy_window(window)
        glfw.terminate()
        
        window = None
        shader = None
        texture = None
        VAO = None

def save_image(image, path):
    if image is not None:
        try:
            image.save(path)
            return True
        except Exception as e:
            print("Lỗi khi lưu ảnh:", e)
    return False

# Lớp chính của ứng dụng
class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Xử lý Ảnh với OpenGL")
        self.root.state('zoomed')  # Đặt cửa sổ ở chế độ tối đa
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_ui()
        self.opengl_initialized = False
        self.current_image = None
        self.thumbnail_label = None
        self.image_size = None

    def setup_ui(self):
        # Tạo giao diện chính với hai phần
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Phần bên trái: hiển thị ảnh và các nút cơ bản
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Khu vực hiển thị ảnh
        self.image_frame = ttk.LabelFrame(self.left_frame, text="Ảnh")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.image_frame, bg="light gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Thanh trạng thái hiển thị kích thước ảnh
        self.status_frame = ttk.Frame(self.left_frame)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Chưa có ảnh nào được tải")
        self.status_label.pack(side=tk.LEFT)
        
        # Các nút cơ bản
        self.button_frame = ttk.Frame(self.left_frame)
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.load_button = ttk.Button(self.button_frame, text="Tải ảnh", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(self.button_frame, text="Lưu ảnh", command=self.save_image)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(self.button_frame, text="Khôi phục gốc", command=self.reset_image)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Phần bên phải: các điều khiển xử lý ảnh
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5, ipadx=10)
        
        # Combobox để chọn chế độ hiển thị
        self.mode_frame = ttk.LabelFrame(self.right_frame, text="Chế độ hiển thị")
        self.mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.mode_var = tk.StringVar(value="Gốc")
        self.mode_combo = ttk.Combobox(self.mode_frame, textvariable=self.mode_var)
        self.mode_combo['values'] = ("Gốc", "Grayscale", "HSV", "CMYK", "RGB", "LAB", "YUV", 
                                    "HSL", "YCbCr", "sRGB", "Adobe RGB", "XYZ", "Negative", "Sepia", "Tùy chỉnh")
        self.mode_combo.pack(fill=tk.X, padx=5, pady=5)
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)
        
        # Khung chứa các điều khiển slider
        self.control_frame = ttk.Frame(self.right_frame)
        self.control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Khởi tạo các tab cho các nhóm điều khiển
        self.tab_control = ttk.Notebook(self.control_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Tab RGB
        self.rgb_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.rgb_tab, text="RGB")
        
        ttk.Label(self.rgb_tab, text="Red").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.red_scale = ttk.Scale(self.rgb_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.red_scale.set(1.0)
        self.red_scale.grid(row=0, column=1, padx=5, pady=5)
        self.red_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("red", self.red_scale.get()))
        
        ttk.Label(self.rgb_tab, text="Green").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.green_scale = ttk.Scale(self.rgb_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.green_scale.set(1.0)
        self.green_scale.grid(row=1, column=1, padx=5, pady=5)
        self.green_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("green", self.green_scale.get()))
        
        ttk.Label(self.rgb_tab, text="Blue").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.blue_scale = ttk.Scale(self.rgb_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.blue_scale.set(1.0)
        self.blue_scale.grid(row=2, column=1, padx=5, pady=5)
        self.blue_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("blue", self.blue_scale.get()))
        
        # Tab HSV
        self.hsv_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.hsv_tab, text="HSV")
        
        ttk.Label(self.hsv_tab, text="Hue").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.hue_scale = ttk.Scale(self.hsv_tab, from_=0, to=360, orient=tk.HORIZONTAL, length=200)
        self.hue_scale.set(0)
        self.hue_scale.grid(row=0, column=1, padx=5, pady=5)
        self.hue_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("hue", self.hue_scale.get()))
        
        ttk.Label(self.hsv_tab, text="Saturation").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.saturation_scale = ttk.Scale(self.hsv_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.saturation_scale.set(1.0)
        self.saturation_scale.grid(row=1, column=1, padx=5, pady=5)
        self.saturation_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("saturation", self.saturation_scale.get()))
        
        ttk.Label(self.hsv_tab, text="Value").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.value_scale = ttk.Scale(self.hsv_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.value_scale.set(1.0)
        self.value_scale.grid(row=2, column=1, padx=5, pady=5)
        self.value_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("value", self.value_scale.get()))
        
        # Tab HSL
        self.hsl_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.hsl_tab, text="HSL")
        
        ttk.Label(self.hsl_tab, text="Lightness").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.lightness_scale = ttk.Scale(self.hsl_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.lightness_scale.set(1.0)
        self.lightness_scale.grid(row=0, column=1, padx=5, pady=5)
        self.lightness_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("lightness", self.lightness_scale.get()))
        
        # Tab CMYK
        self.cmyk_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.cmyk_tab, text="CMYK")
        
        ttk.Label(self.cmyk_tab, text="Cyan").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.cyan_scale = ttk.Scale(self.cmyk_tab, from_=-1, to=1, orient=tk.HORIZONTAL, length=200)
        self.cyan_scale.set(0)
        self.cyan_scale.grid(row=0, column=1, padx=5, pady=5)
        self.cyan_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("cyan", self.cyan_scale.get()))
        
        ttk.Label(self.cmyk_tab, text="Magenta").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.magenta_scale = ttk.Scale(self.cmyk_tab, from_=-1, to=1, orient=tk.HORIZONTAL, length=200)
        self.magenta_scale.set(0)
        self.magenta_scale.grid(row=1, column=1, padx=5, pady=5)
        self.magenta_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("magenta", self.magenta_scale.get()))
        
        ttk.Label(self.cmyk_tab, text="Yellow").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.yellow_scale = ttk.Scale(self.cmyk_tab, from_=-1, to=1, orient=tk.HORIZONTAL, length=200)
        self.yellow_scale.set(0)
        self.yellow_scale.grid(row=2, column=1, padx=5, pady=5)
        self.yellow_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("yellow", self.yellow_scale.get()))
        
        ttk.Label(self.cmyk_tab, text="Key (Black)").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.key_scale = ttk.Scale(self.cmyk_tab, from_=-1, to=1, orient=tk.HORIZONTAL, length=200)
        self.key_scale.set(0)
        self.key_scale.grid(row=3, column=1, padx=5, pady=5)
        self.key_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("key", self.key_scale.get()))
        
        # Tab YUV
        self.yuv_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.yuv_tab, text="YUV")
        
        ttk.Label(self.yuv_tab, text="Y (Luma)").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.y_scale = ttk.Scale(self.yuv_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.y_scale.set(1.0)
        self.y_scale.grid(row=0, column=1, padx=5, pady=5)
        self.y_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("y", self.y_scale.get()))
        
        ttk.Label(self.yuv_tab, text="U").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.u_scale = ttk.Scale(self.yuv_tab, from_=-0.5, to=0.5, orient=tk.HORIZONTAL, length=200)
        self.u_scale.set(0)
        self.u_scale.grid(row=1, column=1, padx=5, pady=5)
        self.u_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("u", self.u_scale.get()))
        
        ttk.Label(self.yuv_tab, text="V").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.v_scale = ttk.Scale(self.yuv_tab, from_=-0.5, to=0.5, orient=tk.HORIZONTAL, length=200)
        self.v_scale.set(0)
        self.v_scale.grid(row=2, column=1, padx=5, pady=5)
        self.v_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("v", self.v_scale.get()))
        
        # Tab YCbCr
        self.ycbcr_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.ycbcr_tab, text="YCbCr")
        
        ttk.Label(self.ycbcr_tab, text="Cb").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.cb_scale = ttk.Scale(self.ycbcr_tab, from_=-0.5, to=0.5, orient=tk.HORIZONTAL, length=200)
        self.cb_scale.set(0)
        self.cb_scale.grid(row=0, column=1, padx=5, pady=5)
        self.cb_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("cb", self.cb_scale.get()))
        
        ttk.Label(self.ycbcr_tab, text="Cr").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.cr_scale = ttk.Scale(self.ycbcr_tab, from_=-0.5, to=0.5, orient=tk.HORIZONTAL, length=200)
        self.cr_scale.set(0)
        self.cr_scale.grid(row=1, column=1, padx=5, pady=5)
        self.cr_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("cr", self.cr_scale.get()))
        
        # Tab LAB
        self.lab_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.lab_tab, text="LAB")
        
        ttk.Label(self.lab_tab, text="L").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.l_scale = ttk.Scale(self.lab_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.l_scale.set(1.0)
        self.l_scale.grid(row=0, column=1, padx=5, pady=5)
        self.l_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("l", self.l_scale.get()))
        
        ttk.Label(self.lab_tab, text="a").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.a_scale = ttk.Scale(self.lab_tab, from_=-1, to=1, orient=tk.HORIZONTAL, length=200)
        self.a_scale.set(0)
        self.a_scale.grid(row=1, column=1, padx=5, pady=5)
        self.a_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("a", self.a_scale.get()))
        
        ttk.Label(self.lab_tab, text="b").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.b_scale = ttk.Scale(self.lab_tab, from_=-1, to=1, orient=tk.HORIZONTAL, length=200)
        self.b_scale.set(0)
        self.b_scale.grid(row=2, column=1, padx=5, pady=5)
        self.b_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("b", self.b_scale.get()))
        
        # Tab XYZ
        self.xyz_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.xyz_tab, text="XYZ")
        
        ttk.Label(self.xyz_tab, text="X").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.x_scale = ttk.Scale(self.xyz_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.x_scale.set(1.0)
        self.x_scale.grid(row=0, column=1, padx=5, pady=5)
        self.x_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("x", self.x_scale.get()))
        
        ttk.Label(self.xyz_tab, text="Y").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.y_xyz_scale = ttk.Scale(self.xyz_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.y_xyz_scale.set(1.0)
        self.y_xyz_scale.grid(row=1, column=1, padx=5, pady=5)
        self.y_xyz_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("y_xyz", self.y_xyz_scale.get()))
        
        ttk.Label(self.xyz_tab, text="Z").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.z_scale = ttk.Scale(self.xyz_tab, from_=0, to=2, orient=tk.HORIZONTAL, length=200)
        self.z_scale.set(1.0)
        self.z_scale.grid(row=2, column=1, padx=5, pady=5)
        self.z_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("z", self.z_scale.get()))
        
        # Tab Sepia
        self.sepia_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.sepia_tab, text="Sepia")
        
        ttk.Label(self.sepia_tab, text="Intensity").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.sepia_scale = ttk.Scale(self.sepia_tab, from_=0, to=1, orient=tk.HORIZONTAL, length=200)
        self.sepia_scale.set(1.0)
        self.sepia_scale.grid(row=0, column=1, padx=5, pady=5)
        self.sepia_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("sepia_intensity", self.sepia_scale.get()))
        
        # Tab điều chỉnh chung
        self.general_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.general_tab, text="Điều chỉnh chung")
        
        ttk.Label(self.general_tab, text="Contrast").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.contrast_scale = ttk.Scale(self.general_tab, from_=0.5, to=2, orient=tk.HORIZONTAL, length=200)
        self.contrast_scale.set(1.0)
        self.contrast_scale.grid(row=0, column=1, padx=5, pady=5)
        self.contrast_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("contrast", self.contrast_scale.get()))
        
        ttk.Label(self.general_tab, text="Brightness").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.brightness_scale = ttk.Scale(self.general_tab, from_=-0.5, to=0.5, orient=tk.HORIZONTAL, length=200)
        self.brightness_scale.set(0)
        self.brightness_scale.grid(row=1, column=1, padx=5, pady=5)
        self.brightness_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("brightness", self.brightness_scale.get()))
        
        ttk.Label(self.general_tab, text="Gamma").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.gamma_scale = ttk.Scale(self.general_tab, from_=0.1, to=3, orient=tk.HORIZONTAL, length=200)
        self.gamma_scale.set(1.0)
        self.gamma_scale.grid(row=2, column=1, padx=5, pady=5)
        self.gamma_scale.bind("<ButtonRelease-1>", lambda e: self.update_filter("gamma", self.gamma_scale.get()))

    def load_image(self):
        global original_image_path
        file_path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            original_image_path = file_path
            self.initialize_opengl(file_path)

    def initialize_opengl(self, image_path):
        """Khởi tạo OpenGL trong thread chính."""
        self.cleanup_current_view()
        result, thumbnail_tk, image_size = setup_opengl(image_path)
        if result:
            self.opengl_initialized = True
            self.setup_image_view(thumbnail_tk, image_size)
            self.update_image()
        else:
            messagebox.showerror("Lỗi", "Không thể khởi tạo OpenGL hoặc tải ảnh.")
            self.opengl_initialized = False

    def setup_image_view(self, thumbnail_tk, image_size):
        """Thiết lập giao diện ban đầu với thumbnail và thông tin kích thước ảnh."""
        if self.thumbnail_label:
            self.thumbnail_label.destroy()
        
        self.thumbnail_label = ttk.Label(self.image_frame, image=thumbnail_tk)
        self.thumbnail_label.image = thumbnail_tk  # Giữ tham chiếu để tránh bị garbage collection
        self.thumbnail_label.pack(pady=10)
        
        self.image_size = image_size
        self.status_label.config(text=f"Kích thước ảnh: {image_size[0]}x{image_size[1]}")

    def cleanup_current_view(self):
        """Dọn dẹp giao diện ảnh hiện tại và tài nguyên OpenGL."""
        if self.thumbnail_label:
            self.thumbnail_label.destroy()
            self.thumbnail_label = None
        if self.opengl_initialized:
            cleanup_opengl()
            self.opengl_initialized = False
        self.current_image = None
        self.status_label.config(text="Chưa có ảnh nào được tải")

    def update_image(self):
        """Cập nhật ảnh chỉ khi được gọi."""
        if not self.opengl_initialized or window is None:
            return
        
        glfw.make_context_current(window)
        processed_image = render_frame()
        
        if processed_image:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = processed_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            processed_image = processed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(processed_image)
            
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.current_image)

    def on_mode_change(self, event):
        """Xử lý thay đổi chế độ từ combobox."""
        global current_mode
        mode_map = {
            "Gốc": MODE_ORIGINAL, "Grayscale": MODE_GRAYSCALE, "HSV": MODE_HSV, "CMYK": MODE_CMYK,
            "RGB": MODE_RGB_ADJUST, "LAB": MODE_LAB, "YUV": MODE_YUV, "HSL": MODE_HSL,
            "YCbCr": MODE_YCBCR, "sRGB": MODE_SRGB, "Adobe RGB": MODE_ADOBE_RGB, "XYZ": MODE_XYZ,
            "Negative": MODE_NEGATIVE, "Sepia": MODE_SEPIA, "Tùy chỉnh": MODE_CUSTOM
        }
        selected_mode = self.mode_var.get()
        current_mode = mode_map.get(selected_mode, MODE_ORIGINAL)
        self.update_image()

    def update_filter(self, key, value):
        """Cập nhật giá trị filter và làm mới ảnh."""
        custom_filter_values[key] = float(value)
        self.update_image()

    def save_image(self):
        """Lưu ảnh đã xử lý hiện tại."""
        if not self.current_image:
            messagebox.showwarning("Cảnh báo", "Không có ảnh để lưu.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if file_path:
            glfw.make_context_current(window)
            processed_image = render_frame()
            if processed_image and save_image(processed_image, file_path):
                messagebox.showinfo("Thành công", "Ảnh đã được lưu thành công!")
            else:
                messagebox.showerror("Lỗi", "Không thể lưu ảnh.")

    def reset_image(self):
        """Đặt lại tất cả filter và chế độ về gốc."""
        global current_mode, custom_filter_values
        current_mode = MODE_ORIGINAL
        
        custom_filter_values = {
            "red": 1.0, "green": 1.0, "blue": 1.0, "hue": 0.0, "saturation": 1.0, "value": 1.0,
            "lightness": 1.0, "cyan": 0.0, "magenta": 0.0, "yellow": 0.0, "key": 0.0,
            "y": 1.0, "u": 0.0, "v": 0.0, "cb": 0.0, "cr": 0.0, "l": 1.0, "a": 0.0, "b": 0.0,
            "contrast": 1.0, "brightness": 0.0, "gamma": 1.0, "sepia_intensity": 1.0,
            "x": 1.0, "y_xyz": 1.0, "z": 1.0,
        }
        
        self.red_scale.set(1.0)
        self.green_scale.set(1.0)
        self.blue_scale.set(1.0)
        self.hue_scale.set(0)
        self.saturation_scale.set(1.0)
        self.value_scale.set(1.0)
        self.lightness_scale.set(1.0)
        self.cyan_scale.set(0)
        self.magenta_scale.set(0)
        self.yellow_scale.set(0)
        self.key_scale.set(0)
        self.y_scale.set(1.0)
        self.u_scale.set(0)
        self.v_scale.set(0)
        self.cb_scale.set(0)
        self.cr_scale.set(0)
        self.l_scale.set(1.0)
        self.a_scale.set(0)
        self.b_scale.set(0)
        self.contrast_scale.set(1.0)
        self.brightness_scale.set(0)
        self.gamma_scale.set(1.0)
        self.sepia_scale.set(1.0)
        self.x_scale.set(1.0)
        self.y_xyz_scale.set(1.0)
        self.z_scale.set(1.0)
        
        self.mode_var.set("Gốc")
        self.update_image()

    def on_closing(self):
        """Xử lý sự kiện đóng cửa sổ."""
        if self.opengl_initialized:
            cleanup_opengl()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()