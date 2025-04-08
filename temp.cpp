#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

enum SeamDirection
{
    VERTICAL,
    HORIZONTAL
};

float energy_image_time = 0;
float cumulative_energy_map_time = 0;
float find_seam_time = 0;
float reduce_time = 0;

bool demo;
bool debug;

Mat readRawImage(const string &filename, int width, int height)
{
    Mat image(height, width, CV_8UC3);
    FILE *file = fopen(filename.c_str(), "rb");
    if (!file)
    {
        cerr << "Error: Unable to open raw image file!" << endl;
        exit(1);
    }

    size_t size = width * height * 3;
    vector<uchar> buffer(size);
    fread(buffer.data(), sizeof(uchar), size, file);
    fclose(file);

    // Copy buffer data to Mat
    memcpy(image.data, buffer.data(), size);
    return image;
}


// Creating Energy map using Gradient magnitude method
Mat createEnergyImage(Mat &image)
{
    clock_t start = clock();
    Mat image_blur, image_gray;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad, energy_image;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // Applying Gaussian blur to reduce noise
    GaussianBlur(image, image_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // Converting to GRAY scale
    cvtColor(image_blur, image_gray, COLOR_BGR2GRAY);

    // Computing gradients in the direction of X and Y
    Scharr(image_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
    Scharr(image_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);

    // Converting gradients to absolute values
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    // Total gradient approx
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // Converting to double precision
    grad.convertTo(energy_image, CV_64F, 1.0 / 255.0);

    // Creating and showing Energy Image
    if (demo)
    {
        namedWindow("Energy Image", WINDOW_AUTOSIZE);
        imshow("Energy Image", energy_image);
    }

    // Calculating time taken to create Energy Image
    clock_t end = clock();
    energy_image_time += ((float)end - (float)start) / CLOCKS_PER_SEC;
    return energy_image;
}


Mat createCumulativeEnergyMap(Mat &energy_image, SeamDirection seam_direction)
{
    clock_t start = clock();
    double a, b, c;

    // Size of Image
    int rowsize = energy_image.rows;
    int colsize = energy_image.cols;

    // Initializing with all zeros
    Mat cumulative_energy_map = Mat(rowsize, colsize, CV_64F, double(0));

    // Copying first row
    if (seam_direction == VERTICAL)
        energy_image.row(0).copyTo(cumulative_energy_map.row(0));
    else if (seam_direction == HORIZONTAL)
        energy_image.col(0).copyTo(cumulative_energy_map.col(0));

    // Taking minimum of three neighbours and adding them all
    if (seam_direction == VERTICAL)
    {
        for (int row = 1; row < rowsize; row++)
        {
            for (int col = 0; col < colsize; col++)
            {
                a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
                b = cumulative_energy_map.at<double>(row - 1, col);
                c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

                cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
            }
        }
    }
    else if (seam_direction == HORIZONTAL)
    {
        for (int col = 1; col < colsize; col++)
        {
            for (int row = 0; row < rowsize; row++)
            {
                a = cumulative_energy_map.at<double>(max(row - 1, 0), col - 1);
                b = cumulative_energy_map.at<double>(row, col - 1);
                c = cumulative_energy_map.at<double>(min(row + 1, rowsize - 1), col - 1);

                cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
            }
        }
    }

    // Creating and showing the newly created cumulative energy map
    if (demo)
    {
        Mat color_cumulative_energy_map;
        double Cmin, Cmax;
        cv::minMaxLoc(cumulative_energy_map, &Cmin, &Cmax);
        float scale = 255.0 / (Cmax - Cmin);
        cumulative_energy_map.convertTo(color_cumulative_energy_map, CV_8UC1, scale);
        applyColorMap(color_cumulative_energy_map, color_cumulative_energy_map, cv::COLORMAP_JET);

        namedWindow("Cumulative Energy Map", WINDOW_AUTOSIZE);
        imshow("Cumulative Energy Map", color_cumulative_energy_map);
    }

    // Calculating time taken to create cumulative energy map
    clock_t end = clock();
    cumulative_energy_map_time += ((float)end - (float)start) / CLOCKS_PER_SEC;
    return cumulative_energy_map;
}


vector<int> findOptimalSeam(Mat &cumulative_energy_map, SeamDirection seam_direction)
{
    clock_t start = clock();
    vector<int> path;
    int rowsize = cumulative_energy_map.rows;
    int colsize = cumulative_energy_map.cols;

    // DP array for tabulation (dynamic programming table)
    Mat dp(rowsize, colsize, CV_64F, Scalar(DBL_MAX)); // Initialize dp array with large values

    if (seam_direction == VERTICAL)
    {
        path.resize(rowsize);

        // Step 1: Fill the first row of the dp array with the energy values from the cumulative_energy_map
        for (int col = 0; col < colsize; col++)
        {
            dp.at<double>(0, col) = cumulative_energy_map.at<double>(0, col);
        }

        // Step 2: Use dynamic programming to calculate the minimum cumulative energy path
        for (int row = 1; row < rowsize; row++)
        {
            for (int col = 0; col < colsize; col++)
            {
                double left = (col > 0) ? dp.at<double>(row - 1, col - 1) : DBL_MAX;
                double up = dp.at<double>(row - 1, col);
                double right = (col < colsize - 1) ? dp.at<double>(row - 1, col + 1) : DBL_MAX;

                dp.at<double>(row, col) = cumulative_energy_map.at<double>(row, col) +
                                           min({left, up, right});
            }
        }

        // Step 3: Find the position of the minimum value in the last row
        double *last_row = dp.ptr<double>(rowsize - 1);
        path[rowsize - 1] = distance(last_row, min_element(last_row, last_row + colsize));

        // Step 4: Backtrack to find the optimal seam
        for (int row = rowsize - 2; row >= 0; row--)
        {
            int col = path[row + 1];
            double *curr_row = dp.ptr<double>(row);

            int offset = 0;
            double min_val = curr_row[col];

            if (col > 0 && curr_row[col - 1] < min_val)
            {
                offset = -1;
                min_val = curr_row[col - 1];
            }
            if (col < colsize - 1 && curr_row[col + 1] < min_val)
            {
                offset = 1;
            }

            path[row] = col + offset;
        }
    }
    else if (seam_direction == HORIZONTAL)
    {
        path.resize(colsize);

        // Step 1: Fill the first column of the dp array with the energy values from the cumulative_energy_map
        for (int row = 0; row < rowsize; row++)
        {
            dp.at<double>(row, 0) = cumulative_energy_map.at<double>(row, 0);
        }

        // Step 2: Use dynamic programming to calculate the minimum cumulative energy path
        for (int col = 1; col < colsize; col++)
        {
            for (int row = 0; row < rowsize; row++)
            {
                double up = (row > 0) ? dp.at<double>(row - 1, col - 1) : DBL_MAX;
                double left = dp.at<double>(row, col - 1);
                double down = (row < rowsize - 1) ? dp.at<double>(row + 1, col - 1) : DBL_MAX;

                dp.at<double>(row, col) = cumulative_energy_map.at<double>(row, col) +
                                           min({up, left, down});
            }
        }

        // Step 3: Find the position of the minimum value in the last column
        double min_val;
        Point min_pt;
        minMaxLoc(dp.col(colsize - 1), &min_val, nullptr, &min_pt, nullptr);
        path[colsize - 1] = min_pt.y;

        // Step 4: Backtrack to find the optimal seam
        for (int col = colsize - 2; col >= 0; col--)
        {
            int row = path[col + 1];
            int offset = 0;

            // Fetch up, left, and down cumulative energies from the previous column
            double up = (row > 0) ? dp.at<double>(row - 1, col) : DBL_MAX;
            double left = dp.at<double>(row, col);
            double down = (row < rowsize - 1) ? dp.at<double>(row + 1, col) : DBL_MAX;

            double min_val = left;

            if (up < min_val)
            {
                offset = -1;
                min_val = up;
            }
            if (down < min_val)
            {
                offset = 1;
            }

            path[col] = row + offset;
        }
    }

    // Calculate time taken
    clock_t end = clock();
    find_seam_time += ((float)end - (float)start) / CLOCKS_PER_SEC;

    return path;
}

void showSeamOnImage(Mat &image, const vector<int> &path, SeamDirection seam_direction)
{
    Mat seam_image = image.clone();

    if (seam_direction == VERTICAL)
    {
        for (int row = 0; row < path.size(); ++row)
        {
            seam_image.at<Vec3b>(row, path[row]) = Vec3b(0, 0, 255);
        }
    }
    else if (seam_direction == HORIZONTAL)
    {
        for (int col = 0; col < path.size(); ++col)
        {
            seam_image.at<Vec3b>(path[col], col) = Vec3b(0, 0, 255);
        }
    }

    namedWindow("Seam Visualization", WINDOW_AUTOSIZE);
    imshow("Seam Visualization", seam_image);
    waitKey(10);
}

void reduce(Mat &image, vector<int> path, SeamDirection seam_direction)
{
    clock_t start = clock();
    int rowsize = image.rows;
    int colsize = image.cols;
    Mat reduced;

    if (seam_direction == VERTICAL)
    {
        reduced = Mat(rowsize, colsize - 1, CV_8UC3);
        for (int row = 0; row < rowsize; row++)
        {
            int col;
            for (col = 0; col < path[row]; col++)
            {
                reduced.at<Vec3b>(row, col) = image.at<Vec3b>(row, col);
            }
            for (col = path[row]; col < colsize - 1; col++)
            {
                reduced.at<Vec3b>(row, col) = image.at<Vec3b>(row, col + 1);
            }
        }
    }
    else if (seam_direction == HORIZONTAL)
    {
        reduced = Mat(rowsize - 1, colsize, CV_8UC3);
        for (int col = 0; col < colsize; col++)
        {
            int row;
            for (row = 0; row < path[col]; row++)
            {
                reduced.at<Vec3b>(row, col) = image.at<Vec3b>(row, col);
            }
            for (row = path[col]; row < rowsize - 1; row++)
            {
                reduced.at<Vec3b>(row, col) = image.at<Vec3b>(row + 1, col);
            }
        }
    }

    image = reduced;

    clock_t end = clock();
    reduce_time += ((float)end - (float)start) / CLOCKS_PER_SEC;
}

void seamCarve(Mat &image, int new_width, int new_height)
{
    if (new_width < image.cols)
    {
        int width_difference = image.cols - new_width;
        for (int i = 0; i < width_difference; i++)
        {
            Mat energy_image = createEnergyImage(image);
            Mat cumulative_energy_map = createCumulativeEnergyMap(energy_image, VERTICAL);
            vector<int> path = findOptimalSeam(cumulative_energy_map, VERTICAL);

            showSeamOnImage(image, path, VERTICAL);

            reduce(image, path, VERTICAL);
        }
    }

    if (new_height < image.rows)
    {
        int height_difference = image.rows - new_height;
        for (int i = 0; i < height_difference; i++)
        {
            Mat energy_image = createEnergyImage(image);
            Mat cumulative_energy_map = createCumulativeEnergyMap(energy_image, HORIZONTAL);
            vector<int> path = findOptimalSeam(cumulative_energy_map, HORIZONTAL);

            showSeamOnImage(image, path, HORIZONTAL);

            reduce(image, path, HORIZONTAL);
        }
    }
}
int main(int argc, char **argv)
{
    string input_filename, output_filename;
    int new_width, new_height;
    int is_raw;
    int width = 0, height = 0;

    cout << "Enter the input image filename: ";
    cin >> input_filename;

    cout << "Is the input file a raw image? (1 for yes, 0 for no): ";
    cin >> is_raw;

    if (is_raw)
    {
        cout << "Enter the width of the raw image: ";
        cin >> width;
        cout << "Enter the height of the raw image: ";
        cin >> height;
    }

    cout << "Enter the new width: ";
    cin >> new_width;
    cout << "Enter the new height: ";
    cin >> new_height;

    cout << "Enter the output image filename: ";
    cin >> output_filename;

    cout << "Enable demo mode (1 for yes, 0 for no): ";
    cin >> demo;

    cout << "Enable debug mode (1 for yes, 0 for no): ";
    cin >> debug;

    Mat image;

    if (is_raw)
    {
        image = readRawImage(input_filename, width, height);
    }
    else
    {
        image = imread(input_filename, IMREAD_COLOR);
        if (!image.data)
        {
            cerr << "Error loading image!" << endl;
            return -1;
        }
    }

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    waitKey(10);

    seamCarve(image, new_width, new_height);

    imwrite(output_filename, image);
    cout << "Output saved to " << output_filename << endl;
    return 0;
    
}
