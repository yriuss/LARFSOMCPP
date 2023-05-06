#ifndef NODE_H
#define NODE_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>


class Node{
private:
    Eigen::VectorXd weight_vector;

    // Depending of the application this can be a structure
    // to add weights to the connections.
    std::vector<uint> edge_connections;
    double a_t;
    uint win_counter = 0;    

    bool check_connections(uint otherNodeIdx);
public:
    Node();
    Node(Eigen::VectorXd weight_vector);
    Node(Eigen::VectorXd weight_vector, std::vector<uint> edge_connections);

    void updateNode(Eigen::VectorXd weight_vector);
    void updateNode(Eigen::VectorXd weight_vector, std::vector<uint> edge_connections);
    void winIncrement();
    uint getWinCounter();
    bool isAlone();
    
    Eigen::VectorXd getWeightVector();
    void addConnection(uint id);
    void removeConnection(uint id);
};

#endif