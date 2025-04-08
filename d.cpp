#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

enum SeamDirection { VERTICAL, HORIZONTAL };

bool demo;
bool debug;
float energy_image_time = 0;
float cumulative_energy_map_time = 0;
float find_seam_time = 0;
float reduce_time = 0;

// Function to read a raw image
Mat readRawImage(const string &filename, int width, int height) {
    Mat image(height, width, CV_8UC3);
    FILE *file = fopen(filename.c_str(), "rb");
    if (!file) {
        cerr << "Error: Unable to open raw image file!" << endl;
        exit(1);
    }
    size_t size = width * height * 3;
    vector<uchar> buffer(size);
    fread(buffer.data(), sizeof(uchar), size, file);
    fclose(file);
    memcpy(image.data, buffer.data(), size);
    return image;
}

// Function to create an energy image using Scharr operator
Mat createEnergyImage(Mat &image) {
    clock_t start = clock();
    Mat image_gray, grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
    GaussianBlur(image, image, Size(3, 3), 0);
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
    Scharr(image_gray, grad_x, CV_32F, 1, 0);
    Scharr(image_gray, grad_y, CV_32F, 0, 1);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    grad.convertTo(grad, CV_32F, 1.0 / 255.0);

    if (demo) {
        imshow("Energy Image", grad);
        waitKey(1);
    }

    energy_image_time += (clock() - start) / (float)CLOCKS_PER_SEC;
    return grad;
}

// Function to create a cumulative energy map
Mat createCumulativeEnergyMap(Mat &energy_image, SeamDirection direction) {
    clock_t start = clock();
    int rows = energy_image.rows, cols = energy_image.cols;
    Mat cumulative_map = energy_image.clone();

    for (int row = 1; row < rows; ++row) {
        float *prev = cumulative_map.ptr<float>(row - 1);
        float *curr = cumulative_map.ptr<float>(row);
        for (int col = 0; col < cols; ++col) {
            float left = (col > 0) ? prev[col - 1] : FLT_MAX;
            float up = prev[col];
            float right = (col < cols - 1) ? prev[col + 1] : FLT_MAX;
            curr[col] += min({left, up, right});
        }
    }

    cumulative_energy_map_time += (clock() - start) / (float)CLOCKS_PER_SEC;
    return cumulative_map;
}

// Function to find the optimal seam
vector<int> findOptimalSeam(Mat &cumulative_map) {
    clock_t start = clock();
    int rows = cumulative_map.rows, cols = cumulative_map.cols;
    vector<int> path(rows);

    int min_index = min_element(cumulative_map.ptr<float>(rows - 1), cumulative_map.ptr<float>(rows - 1) + cols) - cumulative_map.ptr<float>(rows - 1);
    path[rows - 1] = min_index;

    for (int row = rows - 2; row >= 0; --row) {
        int col = path[row + 1];
        float left = (col > 0) ? cumulative_map.at<float>(row, col - 1) : FLT_MAX;
        float up = cumulative_map.at<float>(row, col);
        float right = (col < cols - 1) ? cumulative_map.at<float>(row, col + 1) : FLT_MAX;
        path[row] = col + (left < up ? (left < right ? -1 : 1) : (up < right ? 0 : 1));
    }

    find_seam_time += (clock() - start) / (float)CLOCKS_PER_SEC;
    return path;
}

// Function to reduce the image by removing the seam
void reduce(Mat &image, vector<int> &path) {
    clock_t start = clock();
    int rows = image.rows, cols = image.cols;
    Mat reduced(rows, cols - 1, CV_8UC3);

    for (int row = 0; row < rows; ++row) {
        memmove(reduced.ptr(row), image.ptr(row), path[row] * 3);
        memmove(reduced.ptr(row) + path[row] * 3, image.ptr(row) + (path[row] + 1) * 3, (cols - path[row] - 1) * 3);
    }

    image = reduced;
    reduce_time += (clock() - start) / (float)CLOCKS_PER_SEC;
}

// Function to perform seam carving
void seamCarve(Mat &image, int new_width) {
    while (image.cols > new_width) {
        Mat energy_image = createEnergyImage(image);
        Mat cumulative_map = createCumulativeEnergyMap(energy_image, VERTICAL);
        vector<int> seam = findOptimalSeam(cumulative_map);
        reduce(image, seam);
    }
}

// Main function
int main(int argc, char **argv) {
    string input_filename, output_filename;
    int new_width;
    int is_raw;
    int width = 0, height = 0;

    cout << "Enter the input image filename: ";
    cin >> input_filename;

    cout << "Is the input file a raw image? (1 for yes, 0 for no): ";
    cin >> is_raw;

    if (is_raw) {
        cout << "Enter the width of the raw image: ";
        cin >> width;
        cout << "Enter the height of the raw image: ";
        cin >> height;
    }

    cout << "Enter the new width: ";
    cin >> new_width;

    cout << "Enter the output image filename (e.g., output.png): ";
    cin >> output_filename;

    cout << "Enable demo mode (1 for yes, 0 for no): ";
    cin >> demo;

    cout << "Enable debug mode (1 for yes, 0 for no): ";
    cin >> debug;

    Mat image;

    if (is_raw) {
        image = readRawImage(input_filename, width, height);
    } else {
        image = imread(input_filename, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Error loading image!" << endl;
            return -1;
        }
    }

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    waitKey(10);

    seamCarve(image, new_width);

    if (!imwrite(output_filename, image)) {
        cerr << "Error: Could not save the image. Check the output filename and format." << endl;
        return -1;
    }

    cout << "Output saved to " << output_filename << endl;
    return 0;
}
