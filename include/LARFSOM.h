#ifndef LARFSOM_H
#define LARFSOM_H

#include "Node.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

#define DEBUG 0

#define RANDOM_INIT 0
#define SEQ_INIT    1
#define INF         9999999999999999999999999.

class LARFSOM{
private:
    /// @brief Nodes present in this system.
    std::vector<Node> nodes;
    bool first_cluster_call = true;
    double max_activation;
    uint max_victory_quantity;
    double rho;
    double epsilon;
    double min_error;

    void create_node();
    void create_node(Eigen::VectorXd weight_vector);
    void create_node(Eigen::VectorXd weight_vector, std::vector<uint> edges_connection);
    void remove_node(uint node_id);

    double run_by_epoch(const Eigen::MatrixXd& data);
    void run_by_batch(const Eigen::MatrixXd& data);
public:
    LARFSOM(double rho, double maxActivation, double epsilon, 
            uint initialNodeQuantity, uint maxVictoryQuantity, double minError);
    void cluster(const Eigen::MatrixXd& data, uint nIterations, uint init = RANDOM_INIT);
    std::vector<Node> getNodes();
};


#endif