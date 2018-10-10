#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include "../../../../../../usr/local/cuda/include/driver_types.h"
#include <unistd.h>
#include <stdint.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

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

int main(int argc, char* argv[])
{
    int size = 32*1024*1024;

    int myRank, nRanks, localRank = 0;

    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    //calculating localRank which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    //each process is using two GPUs
    //int nDev = 2;
    int nDev = 1;

    float **h_sendbuff = (float **)malloc(nDev * sizeof(float*));
    float **h_recvbuff = (float **)malloc(nDev * sizeof(float*));

    float** d_sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** d_recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

    //picking GPUs based on localRank
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(localRank*nDev + i));
        CUDACHECK(cudaMalloc(d_sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(d_recvbuff + i, size * sizeof(float)));
        //CUDACHECK(cudaMemset(d_sendbuff[i], 1, size * sizeof(float)));
        //CUDACHECK(cudaMemset(d_recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaHostAlloc(h_sendbuff + i, size * sizeof(float), cudaHostAllocDefault));
        CUDACHECK(cudaHostAlloc(h_recvbuff + i, size * sizeof(float), cudaHostAllocDefault));
        for(int j = 0; j < size; j++) {
            h_sendbuff[i][j] = 10.0f;
            h_recvbuff[i][j] = 1.0f;
        }
        for(int j = 0; j < 10; j++) {
            printf("Send[%i]: %f\n", j, h_sendbuff[i][j]);
            printf("Recv[%i]: %f\n", j, h_recvbuff[i][j]);
        }
        CUDACHECK(cudaMemcpy(d_sendbuff[i], h_sendbuff[i], size * sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_recvbuff[i], h_recvbuff[i], size * sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaStreamCreate(s+i));
    }

    ncclUniqueId id;
    ncclComm_t comms[nDev];

    //generating NCCL unique ID at one process and broadcasting it to all
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    //initializing NCCL, group API is required around ncclCommInitRank as it is
    //called across multiple GPUs in each thread/process
    NCCLCHECK(ncclGroupStart());
    for (int i=0; i<nDev; i++) {
        CUDACHECK(cudaSetDevice(localRank*nDev + i));
        NCCLCHECK(ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i));
    }
    NCCLCHECK(ncclGroupEnd());

    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread/process
    NCCLCHECK(ncclGroupStart());
    for (int i=0; i<nDev; i++)
        NCCLCHECK(ncclAllReduce((const void*)d_sendbuff[i], (void*)d_recvbuff[i], size, ncclFloat, ncclSum,
                                comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA stream to complete NCCL communication
    for (int i=0; i<nDev; i++)
        CUDACHECK(cudaStreamSynchronize(s[i]));

    //copy back
    for(int i = 0; i < nDev; i++) {
        CUDACHECK(cudaMemcpy(h_recvbuff[i], d_recvbuff[i], size  * sizeof(float), cudaMemcpyDeviceToHost));
        //CUDACHECK(cudaMemcpy(h_sendbuff[i], d_sendbuff[i], size  * sizeof(float), cudaMemcpyDeviceToHost));
        for(int j = 0; j < 10; j++) {
            printf("Received: %f\n", h_recvbuff[i][j]);
        }
    }

    //freeing device memory
    for (int i=0; i<nDev; i++) {
        CUDACHECK(cudaFree(d_sendbuff[i]));
        CUDACHECK(cudaFree(d_recvbuff[i]));
        CUDACHECK(cudaFreeHost(h_sendbuff[i]));
        CUDACHECK(cudaFreeHost(h_recvbuff[i]));
    }

    free(d_sendbuff);
    free(d_recvbuff);
    free(h_recvbuff);
    free(h_sendbuff);

    //finalizing NCCL
    for (int i=0; i<nDev; i++) {
        ncclCommDestroy(comms[i]);
    }

    //finalizing MPI
    MPICHECK(MPI_Finalize());

    printf("[MPI Rank %d] Success \n", myRank);
    return 0;
}