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

double euclidean_distance(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
    double sum = 0;
    for (int i = 0; i < v1.size(); i++) {
        sum += pow(v1[i] - v2[i], 2);
    }
    return sqrt(sum);
}

Eigen::MatrixXd matFromCV(const cv::Mat& img) {
    int rows = img.rows;
    int cols = img.cols;
    Eigen::MatrixXd mat(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat(i, j) = img.at<uint8_t>(i, j);
        }
    }

    return mat;
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
    Eigen::MatrixXd outData = matFromCV(reshapedImg);

    // Setting parameters
    double pho = 0.06;
    double maxActivation = 3;
    double epsilon = 0.4;
    double initialNodeQuantity = 2;
    double maxVictoryQuantity = 100;
    double minError = 1e-4;
    
    LARFSOM larfsom(
        pho, 
        maxActivation, 
        epsilon, 
        initialNodeQuantity, 
        maxVictoryQuantity,
        minError
    );
    
    double maxElement = data.maxCoeff();
    double minElement = data.minCoeff();
    
    for(size_t i= 0; i < data.rows(); i++){
        for(size_t j = 0; j <data.cols(); j++){
            data(i,j) = (data(i,j) - minElement)/(maxElement-minElement);
        }
    }
    
    for(size_t i= 0; i < outData.rows(); i++){
        for(size_t j = 0; j <outData.cols(); j++){
            outData(i,j) = (outData(i,j) - minElement)/(maxElement-minElement);
        }
    }


    //data/=maxElement;
    //outData/=maxElement;


    auto start_time = std::chrono::high_resolution_clock::now();

    larfsom.cluster(data, 10, RANDOM_INIT);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;


    std::vector<Node> normalized_segmented_colors = larfsom.getNodes();


    std::cout << "Number of nodes is: " << normalized_segmented_colors.size() << std::endl;

    
    
    // PS*: The last steps in the original LARFSOM paper are different from here,
    // that is, here it's taken a more direct approach, but the system implemented here stills the same.
    for(size_t i = 0; i < outData.rows();i++)
    {
        std::vector<double> dist;
        
        for(size_t j = 0; j < normalized_segmented_colors.size(); j++)
            dist.push_back(euclidean_distance(normalized_segmented_colors[j].getWeightVector().transpose(), outData.row(i)));
        
        auto min_iter = std::min_element(dist.begin(), dist.end());
        int min_index = std::distance(dist.begin(), min_iter);

        //std::cout << min_index << std::endl;
        Eigen::VectorXd toOutDataRow =normalized_segmented_colors[min_index].getWeightVector();

        for(size_t i= 0; i < normalized_segmented_colors[min_index].getWeightVector().rows(); i++){
            toOutDataRow(i) = (normalized_segmented_colors[min_index].getWeightVector()(i)*(maxElement-minElement)+minElement);
        }
        outData.row(i) << toOutDataRow.transpose();
        
    }
    

    //std::cout << outData << std::endl;
    cv::Mat outputMat = cv::Mat(img.rows, img.cols, CV_8UC3); // create output matrix
    
    // copy data from input matrix to output matrix
    for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            outputMat.at<cv::Vec3b>(i,j) = cv::Vec3b((uint)outData(i*img.rows+j,0), (uint)outData(i*img.rows+j,1), (uint)outData(i*img.rows+j,2));
        }
    }


    cv::imshow("Output Image", outputMat);

    cv::waitKey(0);

    return 0;
}