# SmartCrop - Image Resizing with Seam Carving

![SmartCrop Demo](test3.jpg)

SmartCrop is an intelligent image resizing tool that uses the seam carving algorithm to resize images while preserving the most important content. Unlike traditional resizing methods that simply scale the image, SmartCrop removes the least important pixels (seams) to achieve the desired dimensions.

## Features

- Intelligent image resizing using seam carving algorithm
- Support for both regular image formats and raw images
- Real-time visualization of the energy map and seam removal process
- Preserves important content while resizing
- Simple command-line interface

## Prerequisites

- OpenCV library
- C++ compiler (g++ recommended)
- CMake (optional, for building)

## Installation

1. Install OpenCV:
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/SmartCrop.git
cd SmartCrop
```

3. Compile the program:
```bash
g++ d.cpp -o smartcrop `pkg-config --cflags --libs opencv4`
```

## Usage

Run the program:
```bash
./smartcrop
```

The program will prompt you for:
1. Input image filename
2. Whether the input is a raw image (1 for yes, 0 for no)
3. For raw images:
   - Width of the image
   - Height of the image
4. Desired new width
5. Output filename (e.g., output.png)
6. Demo mode (1 for yes, 0 for no) - Shows energy map visualization
7. Debug mode (1 for yes, 0 for no) - Shows additional information

## Example

```bash
./smartcrop
Enter the input image filename: test1.jpg
Is the input file a raw image? (1 for yes, 0 for no): 0
Enter the new width: 800
Enter the output image filename: output.png
Enable demo mode (1 for yes, 0 for no): 1
Enable debug mode (1 for yes, 0 for no): 0
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- Raw image files

## How It Works

1. The program calculates an energy map of the image using the Scharr operator
2. It finds the optimal seam (path of least energy) to remove
3. The seam is removed, and the image is resized
4. This process repeats until the desired width is achieved

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on the seam carving algorithm by Shai Avidan and Ariel Shamir
- Uses OpenCV for image processing 