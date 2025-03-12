#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // Nếu không có tham số, dùng đường dẫn mặc định
    string imgPath;
    imgPath = "C:/Users/DELL/Pictures/anh.jpg"; // Đường dẫn ảnh
    // Tải ảnh (OpenCV sử dụng không gian màu BGR mặc định)
    Mat bgrImage = imread(imgPath,1);
    if (bgrImage.empty())
    {
        cout << "Không thể mở/ tìm thấy file này: " << imgPath << endl;
        return -1;
    }

    // Chuyển đổi sang các không gian màu khác
    Mat hsvImage, labImage, ycrcbImage, grayImage;
    cvtColor(bgrImage, hsvImage, COLOR_BGR2HSV);
    cvtColor(bgrImage, labImage, COLOR_BGR2Lab);
    cvtColor(bgrImage, ycrcbImage, COLOR_BGR2YCrCb);
	cvtColor(bgrImage, grayImage, COLOR_BGR2GRAY);

    // Hiển thị các ảnh với tiêu đề tương ứng
    namedWindow("Original (BGR)", WINDOW_AUTOSIZE);
    imshow("Original (BGR)", bgrImage);

    namedWindow("HSV", WINDOW_AUTOSIZE);
    imshow("HSV", hsvImage);

    namedWindow("Lab", WINDOW_AUTOSIZE);
    imshow("Lab", labImage);

    namedWindow("YCrCb", WINDOW_AUTOSIZE);
    imshow("YCrCb", ycrcbImage);

	namedWindow("Gray", WINDOW_AUTOSIZE);
	imshow("Gray", grayImage);

    int c = waitKey(0);
	if (c == 27) // Ký tự ESC
	{
		destroyAllWindows();
	}
    return 0;
}
