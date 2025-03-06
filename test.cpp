#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

// Tiền khai báo các lớp (để hỗ trợ chuyển đổi qua lại)
class CMYK;
class YIQ;
class HSV;
class HSL;
class XYZ;
class Lab;
class LCH;
class YUV;
class YCbCr;
class ICtCp;

// Lớp RGB đại diện theo chuẩn sRGB (gamma nén)
class RGB {
public:
    double r, g, b; // Giá trị chuẩn hoá [0,1]
    
    RGB(double _r = 0, double _g = 0, double _b = 0) : r(_r), g(_g), b(_b) {}
    
    // Các phương thức chuyển đổi từ RGB sang các không gian màu khác
    CMYK toCMYK() const;
    YIQ toYIQ() const;
    HSV toHSV() const;
    HSL toHSL() const;
    XYZ toXYZ() const;  // Sử dụng RGB tuyến tính (sau khi giải nén gamma)
    Lab toLab() const;  // Qua XYZ
    ICtCp toICtCp() const; // Phiên bản đơn giản (stub)
    YUV toYUV() const;
    
    // Các hàm xử lý gamma cho sRGB
    RGB toLinear_sRGB() const;
    RGB linearTo_sRGB() const;
    
    // Adobe RGB (giả sử gamma ~2.2)
    RGB toLinear_AdobeRGB() const;
    RGB linearTo_AdobeRGB() const;
};

// Các lớp khác

class CMYK {
public:
    double c, m, y, k;
    CMYK(double _c = 0, double _m = 0, double _y = 0, double _k = 0)
        : c(_c), m(_m), y(_y), k(_k) {}
        
    RGB toRGB() const;
};

class YIQ {
public:
    double y, i, q;
    YIQ(double _y = 0, double _i = 0, double _q = 0)
        : y(_y), i(_i), q(_q) {}
        
    RGB toRGB() const;
};

class HSV {
public:
    double h, s, v;
    HSV(double _h = 0, double _s = 0, double _v = 0)
        : h(_h), s(_s), v(_v) {}
        
    RGB toRGB() const;
};

class HSL {
public:
    double h, s, l;
    HSL(double _h = 0, double _s = 0, double _l = 0)
        : h(_h), s(_s), l(_l) {}
        
    RGB toRGB() const;
};

class XYZ {
public:
    double x, y, z;
    XYZ(double _x = 0, double _y = 0, double _z = 0)
        : x(_x), y(_y), z(_z) {}
        
    RGB toRGB() const;
};

class Lab {
public:
    double L, a, b;
    Lab(double _L = 0, double _a = 0, double _b = 0)
        : L(_L), a(_a), b(_b) {}
        
    XYZ toXYZ() const;
    RGB toRGB() const; // Qua XYZ
};

class LCH {
public:
    double L, C, H;
    LCH(double _L = 0, double _C = 0, double _H = 0)
        : L(_L), C(_C), H(_H) {}
        
    // Hàm chuyển đổi: LCH -> Lab -> RGB
    Lab toLab() const;
};

class YUV {
public:
    double y, u, v;
    YUV(double _y = 0, double _u = 0, double _v = 0)
        : y(_y), u(_u), v(_v) {}
        
    RGB toRGB() const;
};

class YCbCr {
public:
    double y, cb, cr;
    YCbCr(double _y = 0, double _cb = 0, double _cr = 0)
        : y(_y), cb(_cb), cr(_cr) {}
        
    RGB toRGB() const;
};

class ICtCp {
public:
    double I, Ct, Cp;
    ICtCp(double _I = 0, double _Ct = 0, double _Cp = 0)
        : I(_I), Ct(_Ct), Cp(_Cp) {}
        
    RGB toRGB() const;
};

//-------------------- TRIỂN KHAI HÀM THÀNH VIÊN --------------------//

// 1. RGB ↔ CMYK
CMYK RGB::toCMYK() const {
    double K = 1 - max({r, g, b});
    double c, m, y;
    if (fabs(K - 1.0) < 1e-6) { // màu đen hoàn toàn
        c = m = y = 0;
    } else {
        c = (1 - r - K) / (1 - K);
        m = (1 - g - K) / (1 - K);
        y = (1 - b - K) / (1 - K);
    }
    return CMYK(c, m, y, K);
}

RGB CMYK::toRGB() const {
    double R = (1 - c) * (1 - k);
    double G = (1 - m) * (1 - k);
    double B = (1 - y) * (1 - k);
    return RGB(R, G, B);
}

// 2. RGB ↔ YIQ
YIQ RGB::toYIQ() const {
    double Y = 0.299 * r + 0.587 * g + 0.114 * b;
    double I = 0.596 * r - 0.274 * g - 0.322 * b;
    double Q = 0.211 * r - 0.523 * g + 0.312 * b;
    return YIQ(Y, I, Q);
}

RGB YIQ::toRGB() const {
    double R = y + 0.956 * i + 0.621 * q;
    double G = y - 0.272 * i - 0.647 * q;
    double B = y - 1.106 * i + 1.703 * q;
    return RGB(R, G, B);
}

// 3. RGB ↔ HSV
HSV RGB::toHSV() const {
    double cmax = max({r, g, b});
    double cmin = min({r, g, b});
    double delta = cmax - cmin;
    double h = 0;
    if (fabs(delta) < 1e-6)
        h = 0;
    else if (cmax == r)
        h = 60 * fmod(((g - b) / delta), 6);
    else if (cmax == g)
        h = 60 * (((b - r) / delta) + 2);
    else // cmax == b
        h = 60 * (((r - g) / delta) + 4);
    if (h < 0) h += 360;
    double s = (cmax == 0) ? 0 : (delta / cmax);
    double v = cmax;
    return HSV(h, s, v);
}

RGB HSV::toRGB() const {
    double C = v * s;
    double X = C * (1 - fabs(fmod(h / 60.0, 2) - 1));
    double m = v - C;
    double r_prime, g_prime, b_prime;
    if (h < 60) {
        r_prime = C; g_prime = X; b_prime = 0;
    } else if (h < 120) {
        r_prime = X; g_prime = C; b_prime = 0;
    } else if (h < 180) {
        r_prime = 0; g_prime = C; b_prime = X;
    } else if (h < 240) {
        r_prime = 0; g_prime = X; b_prime = C;
    } else if (h < 300) {
        r_prime = X; g_prime = 0; b_prime = C;
    } else {
        r_prime = C; g_prime = 0; b_prime = X;
    }
    return RGB(r_prime + m, g_prime + m, b_prime + m);
}

// 4. RGB ↔ HSL
HSL RGB::toHSL() const {
    double cmax = max({r, g, b});
    double cmin = min({r, g, b});
    double delta = cmax - cmin;
    double l = (cmax + cmin) / 2.0;
    double h = 0, s = 0;
    if (fabs(delta) < 1e-6) {
        h = 0;
        s = 0;
    } else {
        s = delta / (1 - fabs(2 * l - 1));
        if (cmax == r)
            h = 60 * fmod(((g - b) / delta), 6);
        else if (cmax == g)
            h = 60 * (((b - r) / delta) + 2);
        else
            h = 60 * (((r - g) / delta) + 4);
        if (h < 0) h += 360;
    }
    return HSL(h, s, l);
}

RGB HSL::toRGB() const {
    double C = (1 - fabs(2 * l - 1)) * s;
    double X = C * (1 - fabs(fmod(h / 60.0, 2) - 1));
    double m = l - C / 2;
    double r_prime, g_prime, b_prime;
    if (h < 60) {
        r_prime = C; g_prime = X; b_prime = 0;
    } else if (h < 120) {
        r_prime = X; g_prime = C; b_prime = 0;
    } else if (h < 180) {
        r_prime = 0; g_prime = C; b_prime = X;
    } else if (h < 240) {
        r_prime = 0; g_prime = X; b_prime = C;
    } else if (h < 300) {
        r_prime = X; g_prime = 0; b_prime = C;
    } else {
        r_prime = C; g_prime = 0; b_prime = X;
    }
    return RGB(r_prime + m, g_prime + m, b_prime + m);
}

// 5. Hàm xử lý gamma cho sRGB: chuyển từ gamma nén sang tuyến tính
RGB RGB::toLinear_sRGB() const {
    auto comp = [](double channel) -> double {
        return (channel <= 0.04045) ? channel / 12.92
                                    : pow((channel + 0.055) / 1.055, 2.4);
    };
    return RGB(comp(r), comp(g), comp(b));
}

// Ngược lại: từ RGB tuyến tính sang sRGB gamma nén
RGB RGB::linearTo_sRGB() const {
    auto comp = [](double channel) -> double {
        return (channel <= 0.0031308) ? 12.92 * channel
                                      : 1.055 * pow(channel, 1.0 / 2.4) - 0.055;
    };
    return RGB(comp(r), comp(g), comp(b));
}

// Adobe RGB (giả sử gamma ~2.2)
RGB RGB::toLinear_AdobeRGB() const {
    return RGB(pow(r, 2.2), pow(g, 2.2), pow(b, 2.2));
}

RGB RGB::linearTo_AdobeRGB() const {
    return RGB(pow(r, 1.0 / 2.2), pow(g, 1.0 / 2.2), pow(b, 1.0 / 2.2));
}

// 6. RGB → XYZ (chuyển trước sang RGB tuyến tính)
XYZ RGB::toXYZ() const {
    RGB lin = this->toLinear_sRGB();
    double X = lin.r * 0.4124564 + lin.g * 0.3575761 + lin.b * 0.1804375;
    double Y = lin.r * 0.2126729 + lin.g * 0.7151522 + lin.b * 0.0721750;
    double Z = lin.r * 0.0193339 + lin.g * 0.1191920 + lin.b * 0.9503041;
    return XYZ(X, Y, Z);
}

RGB XYZ::toRGB() const {
    double r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
    double g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    double b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
    RGB lin(r_lin, g_lin, b_lin);
    return lin.linearTo_sRGB();
}

// 7. Chuyển đổi giữa XYZ và Lab (theo CIE, dùng điểm trắng D65)
static double f_xyz(double t) {
    return (t > 0.008856) ? cbrt(t) : 7.787 * t + 16.0 / 116;
}
static double f_inv(double t) {
    double t3 = t * t * t;
    return (t3 > 0.008856) ? t3 : (t - 16.0 / 116) / 7.787;
}

Lab XYZ_to_Lab(const XYZ &xyz) {
    double Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;
    double fx = f_xyz(xyz.x / Xn);
    double fy = f_xyz(xyz.y / Yn);
    double fz = f_xyz(xyz.z / Zn);
    double L = 116 * fy - 16;
    double a = 500 * (fx - fy);
    double b = 200 * (fy - fz);
    return Lab(L, a, b);
}

XYZ Lab_to_XYZ(const Lab &lab) {
    double Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;
    double fy = (lab.L + 16) / 116.0;
    double fx = lab.a / 500.0 + fy;
    double fz = fy - lab.b / 200.0;
    double X = Xn * f_inv(fx);
    double Y = Yn * f_inv(fy);
    double Z = Zn * f_inv(fz);
    return XYZ(X, Y, Z);
}

Lab RGB::toLab() const {
    XYZ xyz = this->toXYZ();
    return XYZ_to_Lab(xyz);
}

XYZ Lab::toXYZ() const {
    return Lab_to_XYZ(*this);
}

RGB Lab::toRGB() const {
    XYZ xyz = this->toXYZ();
    return xyz.toRGB();
}

// 8. Lab ↔ LCH
LCH Lab_to_LCH(const Lab &lab) {
    double C = sqrt(lab.a * lab.a + lab.b * lab.b);
    double H = atan2(lab.b, lab.a) * 180.0 / M_PI;
    if (H < 0)
        H += 360;
    return LCH(lab.L, C, H);
}

Lab LCH_to_Lab(const LCH &lch) {
    double rad = lch.H * M_PI / 180.0;
    double a = lch.C * cos(rad);
    double b = lch.C * sin(rad);
    return Lab(lch.L, a, b);
}

Lab LCH::toLab() const {
    return LCH_to_Lab(*this);
}

// 9. RGB ↔ YUV (theo BT.601)
YUV RGB::toYUV() const {
    double Y = 0.299 * r + 0.587 * g + 0.114 * b;
    double U = -0.14713 * r - 0.28886 * g + 0.436 * b;
    double V = 0.615 * r - 0.51499 * g - 0.10001 * b;
    return YUV(Y, U, V);
}

RGB YUV::toRGB() const {
    double R = y + 1.13983 * v;
    double G = y - 0.39465 * u - 0.58060 * v;
    double B = y + 2.03211 * u;
    return RGB(R, G, B);
}

// 10. RGB ↔ YCbCr (normalized, giá trị [0,1])
YCbCr RGB_to_YCbCr_normalized(const RGB &rgb) {
    double R = rgb.r, G = rgb.g, B = rgb.b;
    double Y  = 0.299 * R + 0.587 * G + 0.114 * B;
    double Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5;
    double Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5;
    return YCbCr(Y, Cb, Cr);
}

RGB YCbCr::toRGB() const {
    double Y = y, Cb = cb, Cr = cr;
    double R = Y + 1.402 * (Cr - 0.5);
    double G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5);
    double B = Y + 1.772 * (Cb - 0.5);
    return RGB(R, G, B);
}

// 11. RGB ↔ ICtCp (phiên bản đơn giản - stub)
ICtCp RGB::toICtCp() const {
    double I = 0.5 * (r + g + b);
    double Ct = r - b;
    double Cp = g - b;
    return ICtCp(I, Ct, Cp);
}

RGB ICtCp::toRGB() const {
    double avg = I;
    double R = avg + Ct / 2;
    double G = avg + Cp / 2;
    double B = avg - (Ct + Cp) / 2;
    return RGB(R, G, B);
}

//-------------------- HÀM HỖ TRỢ CHUYỂN ĐỔI GIỮA CÁC MÔ HÌNH --------------------//

// Danh sách mã cho các mô hình màu:
// 1: RGB
// 2: CMYK
// 3: YIQ
// 4: HSV
// 5: HSL
// 6: Lab
// 7: LCH
// 8: YUV
// 9: YCbCr
// 10: Grayscale
// 11: XYZ
// 12: ICtCp
// 13: sRGB (nhập như RGB)
// 14: Adobe RGB

// Hàm nhập màu theo mô hình nguồn và chuyển về RGB (chuẩn sRGB gamma nén)
RGB inputColorFromModel(int modelChoice) {
    double a, b, c, d;
    switch (modelChoice) {
        case 1: // RGB
        case 13: { // sRGB
            cout << "Nhap gia tri R, G, B (0 -> 1): ";
            cin >> a >> b >> c;
            return RGB(a, b, c);
        }
        case 2: { // CMYK
            cout << "Nhap gia tri C, M, Y, K (0 -> 1): ";
            cin >> a >> b >> c >> d;
            return CMYK(a, b, c, d).toRGB();
        }
        case 3: { // YIQ
            cout << "Nhap gia tri Y, I, Q: ";
            cin >> a >> b >> c;
            return YIQ(a, b, c).toRGB();
        }
        case 4: { // HSV
            cout << "Nhap gia tri H (0->360), S, V (0->1): ";
            cin >> a >> b >> c;
            return HSV(a, b, c).toRGB();
        }
        case 5: { // HSL
            cout << "Nhap gia tri H (0->360), S, L (0->1): ";
            cin >> a >> b >> c;
            return HSL(a, b, c).toRGB();
        }
        case 6: { // Lab
            cout << "Nhap gia tri L, a, b: ";
            cin >> a >> b >> c;
            return Lab(a, b, c).toRGB();
        }
        case 7: { // LCH
            cout << "Nhap gia tri L, C, H (0->360): ";
            cin >> a >> b >> c;
            return LCH(a, b, c).toLab().toRGB();
        }
        case 8: { // YUV
            cout << "Nhap gia tri Y, U, V: ";
            cin >> a >> b >> c;
            return YUV(a, b, c).toRGB();
        }
        case 9: { // YCbCr
            cout << "Nhap gia tri Y, Cb, Cr (0->1): ";
            cin >> a >> b >> c;
            return YCbCr(a, b, c).toRGB();
        }
        case 10: { // Grayscale
            cout << "Nhap gia tri Grayscale (0->1): ";
            cin >> a;
            return RGB(a, a, a);
        }
        case 11: { // XYZ
            cout << "Nhap gia tri X, Y, Z: ";
            cin >> a >> b >> c;
            return XYZ(a, b, c).toRGB();
        }
        case 12: { // ICtCp
            cout << "Nhap gia tri I, Ct, Cp: ";
            cin >> a >> b >> c;
            return ICtCp(a, b, c).toRGB();
        }
        case 14: { // Adobe RGB
            cout << "Nhap gia tri R, G, B theo Adobe RGB (0->1): ";
            cin >> a >> b >> c;
            // Chuyển từ Adobe RGB gamma sang RGB sRGB bằng cách:
            RGB adobe(a, b, c);
            RGB linearAdb = adobe.toLinear_AdobeRGB();
            // Giả sử nội bộ chúng ta dùng sRGB, chuyển từ tuyến tính sang sRGB:
            return linearAdb.linearTo_sRGB();
        }
        default:
            return RGB();
    }
}

// Hàm chuyển đổi từ RGB (chuẩn sRGB) sang mô hình màu đích và in kết quả
void outputColorToModel(int modelChoice, const RGB &color) {
    switch (modelChoice) {
        case 1: // RGB
        case 13: { // sRGB
            cout << "RGB: (" << color.r << ", " << color.g << ", " << color.b << ")\n";
            break;
        }
        case 2: { // CMYK
            CMYK cmyk = color.toCMYK();
            cout << "CMYK: (" << cmyk.c << ", " << cmyk.m << ", " << cmyk.y << ", " << cmyk.k << ")\n";
            break;
        }
        case 3: { // YIQ
            YIQ yiq = color.toYIQ();
            cout << "YIQ: (" << yiq.y << ", " << yiq.i << ", " << yiq.q << ")\n";
            break;
        }
        case 4: { // HSV
            HSV hsv = color.toHSV();
            cout << "HSV: (" << hsv.h << ", " << hsv.s << ", " << hsv.v << ")\n";
            break;
        }
        case 5: { // HSL
            HSL hsl = color.toHSL();
            cout << "HSL: (" << hsl.h << ", " << hsl.s << ", " << hsl.l << ")\n";
            break;
        }
        case 6: { // Lab
            Lab lab = color.toLab();
            cout << "Lab: (" << lab.L << ", " << lab.a << ", " << lab.b << ")\n";
            break;
        }
        case 7: { // LCH
            Lab lab = color.toLab();
            LCH lch = Lab_to_LCH(lab);
            cout << "LCH: (" << lch.L << ", " << lch.C << ", " << lch.H << ")\n";
            break;
        }
        case 8: { // YUV
            YUV yuv = color.toYUV();
            cout << "YUV: (" << yuv.y << ", " << yuv.u << ", " << yuv.v << ")\n";
            break;
        }
        case 9: { // YCbCr
            YCbCr ycbcr = RGB_to_YCbCr_normalized(color);
            cout << "YCbCr (normalized): (" << ycbcr.y << ", " << ycbcr.cb << ", " << ycbcr.cr << ")\n";
            break;
        }
        case 10: { // Grayscale
            double gray = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
            cout << "Grayscale: " << gray << "\n";
            break;
        }
        case 11: { // XYZ
            XYZ xyz = color.toXYZ();
            cout << "XYZ: (" << xyz.x << ", " << xyz.y << ", " << xyz.z << ")\n";
            break;
        }
        case 12: { // ICtCp
            ICtCp ictcp = color.toICtCp();
            cout << "ICtCp (stub): (" << ictcp.I << ", " << ictcp.Ct << ", " << ictcp.Cp << ")\n";
            break;
        }
        case 14: { // Adobe RGB
            // Chuyển từ sRGB (nội bộ) sang Adobe RGB:
            RGB linear = color.toLinear_sRGB();
            RGB adobe = linear.linearTo_AdobeRGB();
            cout << "Adobe RGB: (" << adobe.r << ", " << adobe.g << ", " << adobe.b << ")\n";
            break;
        }
        default:
            cout << "Khong ro model nay.\n";
    }
}

//-------------------- GIAO DIỆN NHẬP LIỆU CHÍNH --------------------//

// Hiển thị danh sách các mô hình màu
void showModelList() {
    cout << " 1: RGB\n";
    cout << " 2: CMYK\n";
    cout << " 3: YIQ\n";
    cout << " 4: HSV\n";
    cout << " 5: HSL\n";
    cout << " 6: Lab\n";
    cout << " 7: LCH\n";
    cout << " 8: YUV\n";
    cout << " 9: YCbCr\n";
    cout << "10: Grayscale\n";
    cout << "11: XYZ\n";
    cout << "12: ICtCp\n";
    cout << "13: sRGB (nhap nhu RGB)\n";
    cout << "14: Adobe RGB\n";
}

int main() {
    int srcModel, tgtModel;
    cout << "---------------- CHUONG TRINH CHUYEN DOI MAU ----------------\n";
    
    // Nhập mô hình màu nguồn
    cout << "Chon mo hinh mau NGUON:\n";
    showModelList();
    cout << "Nhap lua chon: ";
    cin >> srcModel;
    
    RGB canonical = inputColorFromModel(srcModel);
    
    // Nhập mô hình màu đích
    cout << "\nChon mo hinh mau DICH:\n";
    showModelList();
    cout << "Nhap lua chon: ";
    cin >> tgtModel;
    
    cout << "\nKET QUA CHUYEN DOI:\n";
    outputColorToModel(tgtModel, canonical);
    
    cout << "-------------------------------------------------------------\n";
    return 0;
}
