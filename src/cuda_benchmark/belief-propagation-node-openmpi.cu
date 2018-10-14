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
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    fflush(out);

    run_test_loopy_belief_propagation_mtx_files_cuda_openmpi("/home/mjt5v/Desktop/gunrock_benchmark_files/10_20.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/10_20.bif.nodes.mtx", out, my_rank, n_ranks, num_devices);

    MPI_Finalize();

    return EXIT_SUCCESS;
}