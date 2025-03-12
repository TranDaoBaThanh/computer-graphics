#include <SFML/Graphics.hpp>
#include <iostream>

int main()
{
    // Nhập màu cho bên trái
    int r1, g1, b1;
    std::cout << "Nhap mau ben trai (R G B, cach nhau boi khoang trang, tu 0 den 255): ";
    std::cin >> r1 >> g1 >> b1;
    sf::Color leftColor(static_cast<sf::Uint8>(r1), static_cast<sf::Uint8>(g1), static_cast<sf::Uint8>(b1));

    // Nhập màu cho bên phải
    int r2, g2, b2;
    std::cout << "Nhap mau ben phai (R G B, cach nhau boi khoang trang, tu 0 den 255): ";
    std::cin >> r2 >> g2 >> b2;
    sf::Color rightColor(static_cast<sf::Uint8>(r2), static_cast<sf::Uint8>(g2), static_cast<sf::Uint8>(b2));

    // Kích thước cửa sổ
    const unsigned int windowWidth = 800;
    const unsigned int windowHeight = 600;
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Color Gradient Generator");
    window.setFramerateLimit(60);

    // Sử dụng VertexArray với 4 đỉnh cho một hình chữ nhật toàn bộ cửa sổ.
    // SFML sẽ tự nội suy màu giữa các đỉnh.
    sf::VertexArray gradientQuad(sf::Quads, 4);

    // Thiết lập vị trí các đỉnh:
    gradientQuad[0].position = sf::Vector2f(0, 0);                   // Trái trên
    gradientQuad[1].position = sf::Vector2f(windowWidth, 0);           // Phải trên
    gradientQuad[2].position = sf::Vector2f(windowWidth, windowHeight);  // Phải dưới
    gradientQuad[3].position = sf::Vector2f(0, windowHeight);          // Trái dưới

    // Gán màu cho các đỉnh:
    // Ở bên trái sử dụng màu leftColor, bên phải sử dụng rightColor.
    gradientQuad[0].color = leftColor;
    gradientQuad[3].color = leftColor;
    gradientQuad[1].color = rightColor;
    gradientQuad[2].color = rightColor;

    while (window.isOpen())
    {
        // Xử lý sự kiện (ví dụ: đóng cửa sổ)
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(gradientQuad);
        window.display();
    }

    return 0;
}
