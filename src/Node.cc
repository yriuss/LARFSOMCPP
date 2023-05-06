#include "Node.h"


Node::Node(){
    
}

Node::Node(Eigen::VectorXd weight_vector){
    this->weight_vector = weight_vector;
}

Node::Node(Eigen::VectorXd weight_vector, std::vector<uint> edge_connections){
    this->weight_vector = weight_vector;
    this->edge_connections = edge_connections;
}

void Node::updateNode(Eigen::VectorXd weight_vector){
    this->weight_vector = weight_vector;
}

void Node::updateNode(Eigen::VectorXd weight_vector, std::vector<uint> edge_connections){
    this->weight_vector = weight_vector;
    this->edge_connections = edge_connections;
}

Eigen::VectorXd Node::getWeightVector(){
    return weight_vector;
}

void Node::winIncrement(){
    win_counter++;
}

uint Node::getWinCounter(){
    return win_counter;
}

bool Node::check_connections(uint id){
    for(size_t i = 0; i < edge_connections.size(); i++){
        if(edge_connections[i] == id)
            return true;
    }
    return false;
}

void Node::removeConnection(uint id){
    bool connection_exist = check_connections(id);
    if(connection_exist)
        return;
    else
        edge_connections.erase(edge_connections.begin() + id);
}

bool Node::isAlone(){
    return edge_connections.empty();
}

void Node::addConnection(uint id){
    bool connection_exist = check_connections(id);
    if(connection_exist)
        return;
    else
        edge_connections.push_back(id);
}