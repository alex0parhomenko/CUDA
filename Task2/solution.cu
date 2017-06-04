#include <iostream>
#include <cuda.h>
#include <string>

using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

__global__ void process(unsigned char* input_img, unsigned char* output_img, int radius, int img_x, int img_y) {
    int img_row = blockIdx.x * blockDim.x  + threadIdx.x;
    int img_col = blockIdx.y * blockDim.y + threadIdx.y;
    if (img_row < img_x && img_col < img_y) {
        int hist[256][3] = {{0}};
        int sum = 0, local_sum_r = 0, local_sum_g = 0, local_sum_b = 0;
        int x_start = max(0, img_row - radius);
        int x_end = min(img_x - 1, img_row + radius);
        int y_start = max(0, img_col - radius);
        int y_end = min(img_y - 1, img_col + radius);
        for (int i = x_start; i <= x_end; i++) {
            for (int j = y_start; j <= y_end; j++) {
                sum++;
                hist[input_img[(i * img_y + j) * 3]][0]++;
                hist[input_img[(i * img_y + j) * 3 + 1]][1]++;
                hist[input_img[(i * img_y + j) * 3 + 2]][2]++; 
            }
        }  
        bool is_r_enable = true, is_g_enable = true, is_b_enable = true;
        for (int i = 0; i < 256; ++i) {
            local_sum_r += hist[i][0];
            local_sum_g += hist[i][1];
            local_sum_b += hist[i][2];
            if (is_r_enable && local_sum_r >= sum / 2) {
                output_img[(img_row * img_y + img_col) * 3] = i;
                is_r_enable = false; 
            }
            if (is_g_enable && local_sum_g >= sum / 2) {
                output_img[(img_row * img_y + img_col) * 3 + 1] = i;
                is_g_enable = false; 
            }
            if (is_b_enable && local_sum_b >= sum / 2) {
                output_img[(img_row * img_y + img_col) * 3 + 2] = i;
                is_b_enable = false;
            }
        }    
    } 

}

int main(int argc, char* argv[]) {
    using namespace cv;

    cv::Mat img_load = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    auto img_size = img_load.size();
    int radius = 0;
    sscanf(argv[2], "%d", &radius);
    int grid_size = 32;
    int rows = img_size.height, cols = img_size.width;
    dim3 grid_dim(grid_size, grid_size);
    dim3 block_dim(rows / grid_size + 1, cols / grid_size + 1);
     
    unsigned char* input_img = new unsigned char[rows * cols * 3];
    unsigned char* output_img = new unsigned char[rows * cols * 3];  
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) { 
            input_img[(i * cols + j) * 3] = img_load.at<cv::Vec3b>(i, j).val[0]; 
            input_img[(i * cols + j) * 3 + 1] = img_load.at<cv::Vec3b>(i, j).val[1];
            input_img[(i * cols + j) * 3 + 2] = img_load.at<cv::Vec3b>(i, j).val[2];
        }
    }
    
    unsigned char* device_input_img;
    unsigned char* device_output_img; 
    cudaMalloc((void**)(&device_input_img), rows *cols * 3 * sizeof(unsigned char));
    cudaMemcpy(device_input_img, input_img, rows*cols*3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc((void**)(&device_output_img), rows*cols *3 *sizeof(unsigned char));
    process<<< grid_dim, block_dim >>>(device_input_img, device_output_img, radius, rows, cols); 
   
    cudaMemcpy(output_img, device_output_img, rows*cols*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //std::cout << img_result.size().height << " " << img_result.size().width << " " << rows << " " << cols << endl; 
    cv::Mat img_result(img_load);
    cv::resize(img_result, img_result, cv::Size(cols, rows)); 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cv::Vec3b new_color(output_img[(i * cols + j) * 3], output_img[(i * cols + j) * 3 + 1], output_img[(i * cols + j) * 3 + 2]);
            //cout << static_cast<int>(output_img[(i * cols + j) * 3]) << " " << static_cast<int>(output_img[(i * cols + j) * 3 + 1]) << " " << static_cast<int>(output_img[(i * cols + j) * 3 + 2]) << " " << i << " " << j << endl;
            //cout << i << " " << j << endl;
            img_result.at<Vec3b>(i, j) = new_color;//cv::Vec3b(1, 1, 1);
        }
    }
    //return 0;
    imwrite( "result_img.jpg", img_result ); 
     
    cudaFree(device_input_img);
    cudaFree(device_output_img);
    delete[] input_img; 
    return 0;
}
