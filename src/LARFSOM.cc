#include "LARFSOM.h"

/// @brief 
/// @param rho 
/// @param maxActivation A activation treshold used as criterion for node creation;
/// @param epsilon 
/// @param initialNodeQuantity Initial number of nodes (minimum is 2);
/// @param maxVictoryQuantity 
/// @param minError A minimum error threshold as a end criterion.
LARFSOM::LARFSOM(double rho, double maxActivation, double epsilon, 
                 uint initialNodeQuantity, uint maxVictoryQuantity, double minError)
{
    for(size_t i = 0; i < initialNodeQuantity; i++)
        create_node();
    
    if(maxActivation < 0)
        throw std::runtime_error("Maximum activation can't be negative.");

    max_activation = maxActivation;
    max_victory_quantity = maxVictoryQuantity;
    this->rho = rho;
    this->epsilon = epsilon;
    min_error = minError;
}

void LARFSOM::create_node(){
    Node node;
    nodes.push_back(node);
}

void LARFSOM::create_node(Eigen::VectorXd weight_vector){
    Node node(weight_vector);
    nodes.push_back(node);
}

void LARFSOM::create_node(Eigen::VectorXd weight_vector,std::vector<uint> edges_connection){
    Node node(weight_vector, edges_connection);
    nodes.push_back(node);
}

void LARFSOM::remove_node(uint node_id){
    nodes.erase(nodes.begin() + node_id);
}

std::vector<Node> LARFSOM::getNodes(){
    return nodes;
}


void LARFSOM::cluster(const Eigen::MatrixXd& data, uint nIterations, uint init){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.rows());

    // In the first call, the data should have at least 2 samples
    // for the nodes initialization.
    if(first_cluster_call && data.rows() < 2)
        throw std::runtime_error("Data is too small!");

    // Also, in the first cluster call there are only 2 empty nodes.
    // Then it's necessary to fill these nodes for the training phase.
    if(first_cluster_call){
        // Selecting 2 samples from the data given an uniform distribution.
        int rand1 = dis(gen);
        int rand2 = dis(gen);

        // From the paper, these samples are going to be the weight vectors
        // of the 2 initial nodes
        nodes[0].updateNode(data.row(rand1));
        nodes[1].updateNode(data.row(rand2));
    }
    
    Eigen::MatrixXd input_data;

    // Setting the input to random or sequential
    if(init == RANDOM_INIT){
        std::vector<int> indices(data.rows());
        std::iota(indices.begin(), indices.end(), 0); // fill the vector with row indices

        std::shuffle(indices.begin(), indices.end(), gen); // shuffle the row indices

        Eigen::MatrixXd shuffledMatrix(data.rows(), data.cols());
        for (int i = 0; i < indices.size(); i++) {
            shuffledMatrix.row(i) = data.row(indices[i]); // reorder the matrix rows
        }
        
        input_data = shuffledMatrix;
    }else if(init == SEQ_INIT)
        input_data = data;
    else
        throw std::runtime_error("Initialization is not valid!\nOnly RANDOM_INIT or SEQ_INIT are valid options.");

    double error;
    for (size_t i = 0; i < nIterations; i++)
    {
#if DEBUG
        std::cout << "Epoch " << i+1 << std::endl;
#endif    
        if(1)
            error = run_by_epoch(input_data);
        else
            run_by_batch(input_data);
        if(error < min_error)
            break;
    }

    first_cluster_call = false;
}

double euclidean_distance(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
    double sum = 0;
    for (int i = 0; i < v1.size(); i++) {
        sum += pow(v1[i] - v2[i], 2);
    }
    return sqrt(sum);
}

double LARFSOM::run_by_epoch(const Eigen::MatrixXd& data){
    Eigen::VectorXd input2nodes_dist;
    uint closest_idx, sec_closest_idx;
    double receptive_field, winner_activation;
    double d_s1_s2, d_s1_to_new_node, d_s2_to_new_node;

    for(size_t i = 0; i < data.rows(); i++)
    {
        
#if DEBUG
        std::cout << "it: " << i << std::endl;
        std::cout << "nodes size: " << nodes.size() << std::endl;
#endif
        //std::cout << "data" << i+1 << std::endl;
        // Finding best matching units...
        
        for(size_t j = 0; j < nodes.size(); j++){
            // Same effect to the VectorXd as a push_back in std::vector
            input2nodes_dist.conservativeResize(input2nodes_dist.size() + 1);
            input2nodes_dist[j] = euclidean_distance(data.row(i), nodes[j].getWeightVector());
        }

        // Getting the winner and updating its distance to the possible new node
        d_s1_to_new_node = input2nodes_dist.minCoeff(&closest_idx);
        
        input2nodes_dist(closest_idx) = INF;


        // Getting the second closest node and updating its distance 
        // to the possible new node.
        d_s2_to_new_node = input2nodes_dist.minCoeff(&sec_closest_idx);

        // Increment the counter of the winner
        nodes[closest_idx].winIncrement();


        // Calculate receptive field
        receptive_field = euclidean_distance(
            nodes[closest_idx].getWeightVector(), 
            nodes[sec_closest_idx].getWeightVector()
        );
#if DEBUG
        std::cout << "closest_idx: " << closest_idx << std::endl;
        std::cout << "sec_closest_idx: " << sec_closest_idx << std::endl;
        std::cout << "receptive_field: " << receptive_field << std::endl;
#endif      
        // If a connection doesn't exist, a new connection is added.
        // The nodes have this logic implemented in its methods.
        nodes[closest_idx].addConnection(sec_closest_idx);
        nodes[sec_closest_idx].addConnection(closest_idx);
        
        
        // Calculate activation of S1
        winner_activation = exp(
            -(euclidean_distance(data.row(i), nodes[closest_idx].getWeightVector()))
        )/receptive_field;
#if DEBUG
        std::cout << "winner activation: " << winner_activation << std::endl;
#endif
        

        // Insert node
        if(winner_activation < max_activation){
            // Remove the connection between S1 and S2
            nodes[closest_idx].removeConnection(sec_closest_idx);
            nodes[sec_closest_idx].removeConnection(closest_idx);

            // Adding a new node with the same weight of current input
            create_node(data.row(i));
            uint idx_new_node = nodes.size() - 1;

            // It is not necessary to recalculate the distances, so
            // in this step only d_s1_s2 is updated
            d_s1_s2 = receptive_field;

            if(d_s1_s2 <= d_s1_to_new_node && d_s2_to_new_node <= d_s1_to_new_node)
            {
                nodes[closest_idx].addConnection(sec_closest_idx);
                nodes[sec_closest_idx].addConnection(closest_idx);
                nodes[idx_new_node].addConnection(sec_closest_idx);
                nodes[sec_closest_idx].addConnection(idx_new_node);
            }
            else if(d_s1_to_new_node <= d_s1_s2 && d_s2_to_new_node <= d_s1_s2)
            {
                nodes[closest_idx].addConnection(idx_new_node);
                nodes[idx_new_node].addConnection(closest_idx);
                nodes[idx_new_node].addConnection(sec_closest_idx);
                nodes[sec_closest_idx].addConnection(idx_new_node);
            }
            else if(d_s1_s2 <= d_s2_to_new_node && d_s1_to_new_node <= d_s2_to_new_node)
            {
                nodes[closest_idx].addConnection(sec_closest_idx);
                nodes[sec_closest_idx].addConnection(closest_idx);
                nodes[idx_new_node].addConnection(closest_idx);
                nodes[closest_idx].addConnection(idx_new_node);
            }
            
        }else{
            uint best_node_win_counter = nodes[closest_idx].getWinCounter();
            Eigen::VectorXd adjust;
            double rho_f;
            
            if(best_node_win_counter < max_victory_quantity)
                rho_f = pow(epsilon*rho, static_cast<double>(best_node_win_counter)/max_victory_quantity);
            else
                rho_f = epsilon*rho;
            adjust = rho_f*(data.row(i) - nodes[closest_idx].getWeightVector().transpose());
            nodes[closest_idx].updateNode(nodes[closest_idx].getWeightVector()+adjust);
        }

        // Clear necessary data for next iteration
        input2nodes_dist.resize(0);        
    }

    // Calculating error
    double error = 0;
        
    // idxs to iterate through the nodes
    uint it1 = nodes.size() - 1;
    uint it2 = it1 - 1;


    Eigen::VectorXd diff_vector;

    for(size_t ii = 0; ii < it2; ii++)
    {
        diff_vector = (nodes[it1].getWeightVector() - nodes[it2].getWeightVector());
        error += diff_vector.squaredNorm();
        it1--;
        it2--;
    }

    error = error/nodes.size();
    
    // Remove disconnected nodes
    for(int i = nodes.size() - 1; i >= 0; i--)
    {
        if(nodes[i].isAlone())
            remove_node(i);
    }

    return error;
}

void LARFSOM::run_by_batch(const Eigen::MatrixXd& data){
    //TODO
}