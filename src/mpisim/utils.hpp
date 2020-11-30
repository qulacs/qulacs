#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <mpi.h>

class DllExport Utility{
    public:
        void static send_vector(int sender,int receiver,int tag,std::vector<UINT> &data){
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
            if(myrank == sender){
                UINT data_size = data.size();
                MPI_Request req;
                MPI_Send(&data_size,1,MPI_UNSIGNED,receiver,tag,MPI_COMM_WORLD);
                MPI_Send(data.data(),data_size,MPI_UNSIGNED,receiver,tag + 1,MPI_COMM_WORLD);
            }
        }
        void static receive_vector(int receiver,int sender,int tag,std::vector<UINT> &data){
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
            if(myrank == receiver){
                UINT data_size;
                MPI_Status stat;
                MPI_Recv(&data_size,1,MPI_UNSIGNED,sender,tag,MPI_COMM_WORLD,&stat);
                data.resize(data_size);
                MPI_Recv(data.data(),data_size,MPI_UNSIGNED,sender,tag+1,MPI_COMM_WORLD,&stat); 
            }
        }
};