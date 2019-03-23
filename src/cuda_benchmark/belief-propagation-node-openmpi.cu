#include "belief-propagation.hpp"

static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

int main(int argc, char *argv[]) {
    int my_rank, n_ranks, local_rank, num_devices = 0;
    struct cudaDeviceProp prop;

    // initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));

    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, 0));

    // calculate local rank
    uint64_t host_hashes[n_ranks];
    char hostname[1024];
    getHostName(hostname, 1024);
    host_hashes[my_rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host_hashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for(int p = 0; p < n_ranks; p++) {
        if (p == my_rank) {
            break;
        }
        if (host_hashes[p] == host_hashes[my_rank]) {
            local_rank++;
        }
    }

    // get the number of devices
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&num_devices));

    // start the fun
    FILE * out = fopen("cuda_benchmark_loopy_node_openmpi.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Max In-degree,Avg In-degree,Max Out-degree,Avg Out-degree,Number of Iterations,BP Run Time(s),Total Run Time(s)\n");
    fflush(out);

    struct joint_probability edge_joint_probability;
    size_t dim_x, dim_y;
    set_joint_probability_yahoo_web(&edge_joint_probability, &dim_x, &dim_y);


    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/10_nodes_40_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/10_nodes_40_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/100_nodes_400_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/100_nodes_400_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/1000_nodes_4000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/1000_nodes_4000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/10000_nodes_40000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/10000_nodes_40000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/100000_nodes_400000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/100000_nodes_400000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/200000_nodes_800000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/200000_nodes_800000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/400000_nodes_1600000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/400000_nodes_1600000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/600000_nodes_2400000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/600000_nodes_2400000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/800000_nodes_3200000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/800000_nodes_3200000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/1000000_nodes_4000000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/1000000_nodes_4000000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/2000000_nodes_8000000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/2000000_nodes_8000000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);

    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-delicious_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-delicious_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-follows-mun_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-follows-mun_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-google-plus_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-google-plus_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/web-Stanford_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/web-Stanford_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/web-it-2004_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/web-it-2004_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/web-wiki-ch-internal.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/web-wiki-ch-internal.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);

    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/kron_g500-logn18_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/kron_g500-logn18_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/tech-p2p_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/tech-p2p_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/kron_g500-logn17_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/kron_g500-logn17_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-orkut_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-orkut_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    //run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);
    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_2_red.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out, my_rank, n_ranks, num_devices);

    MPI_Finalize();

    return EXIT_SUCCESS;
}