#include <LARFSOM.h>

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

Eigen::MatrixXd vectorToMatrix(const std::vector<std::vector<uint8_t>>& vec) {
    int rows = vec.size();
    int cols = vec[0].size();
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat(i, j) = vec[i][j];
        }
    }
    return mat;
}

std::vector<std::vector<uint8_t>> matToVector(const cv::Mat& img) {
    std::vector<std::vector<uint8_t>> imgVec;
    for (int i = 0; i < img.rows; i++) {
        std::vector<uint8_t> rowVec;
        for (int j = 0; j < img.cols; j++) {
            rowVec.push_back(img.at<uint8_t>(i, j));
        }
        imgVec.push_back(rowVec);
    }
    return imgVec;
}

bool compareRows(const std::vector<uint8_t>& row1, const std::vector<uint8_t>& row2) {
    // Compare rows element-wise
    for (int i = 0; i < row1.size(); i++) {
        if (row1[i] != row2[i]) {
            return row1[i] < row2[i];
        }
    }
    return false;
}

std::vector<std::vector<uint8_t>> getUniqueColors(cv::Mat& img){
    std::vector<std::vector<uint8_t>> imgRows = matToVector(img);
    
    // Sort the vector
    std::sort(imgRows.begin(), imgRows.end(), compareRows);

    // Use std::unique to find the unique elements in the vector
    auto it = std::unique(imgRows.begin(), imgRows.end());

    // Resize the vector to remove the duplicates
    imgRows.resize(std::distance(imgRows.begin(), it));

    return imgRows;
}

int main(){
    // read an example image
    std::string fileName = "../examples/color_segmentation/house.tiff";
    cv::Mat img = cv::imread(fileName);
    
    cv::Mat reshapedImg = img.reshape(1, img.rows * img.cols);

    // show input image
    cv::imshow("Input Image", img);

    // removing repeated colors    
    Eigen::MatrixXd data = vectorToMatrix(getUniqueColors(reshapedImg));


    // Setting parameters
    double pho = 0.06;
    double maxActivation = 5;
    double epsilon = 0.4;
    double initialNodeQuantity = 2;
    double maxVictoryQuantity = 100;
    double minError = 1e-04;
    std::cout << "Minimum error: " << minError << std::endl;

    LARFSOM larfsom(
        pho, 
        maxActivation, 
        epsilon, 
        initialNodeQuantity, 
        maxVictoryQuantity,
        minError
    );
    
    larfsom.cluster(data, 100, RANDOM_INIT);

    cv::waitKey(0);

    return 0;
}