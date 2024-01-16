////////////////////////// main_receiver.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#define  BUFF_SIZE   2048

#include "type_definitions.h"
typedef struct {
  long data_type;
  int  data_num;
  char data_buff[2048];
} message_seed_t;

int main(void){
	message_seed_t   data;
	int      msqid;
	if ( -1 == ( msqid = msgget( (key_t)1234, IPC_CREAT | 0666)))
	{
		perror( "msgget() failed");
		exit( 1);
	}
	 int        msg = 200;
	// send the request (empty message)
	memcpy(data.data_buff, &msg, sizeof(int));
	data.data_type = TYPE_REQUEST;

	int snd_status = msgsnd(msqid, &data, 0, 0);
	if (snd_status == -1) {
		perror("request send failed");
	}

	while(1)
	{
		// datatype 0: receive all datatype
		// change datatype to 1 if you want to receive python data
		if (msgrcv( msqid, &data, sizeof(message_seed_t) - sizeof( long), 0, 0)==-1)
		{
			printf("error\n");
			perror( "msgrcv() failed");
			exit( 1);
		}
		printf("*** New message received ***\nRaw data: ");
		printf("\n\nRECEIVE SEED %ld is: %s\n",strlen(data.data_buff),data.data_buff);
	}
}

// typedef struct {
// 	long  data_type;
// 	int   data_num;
// 	unsigned char  data_buff[BUFF_SIZE];
// } t_data;

// int main( void)
// {
// 	int      msqid;
// 	t_data   data;
// 	int dval;
// 	double   dValue1, dValue2;
// 	char	 new_array[BUFF_SIZE];

// 	if ( -1 == ( msqid = msgget( (key_t)1234, IPC_CREAT | 0666)))
// 	{
// 		perror( "msgget() failed");
// 		exit( 1);
// 	}

// 	while( 1 )
// 	{
// 		// datatype 0: receive all datatype
// 		// change datatype to 1 if you want to receive python data
// 		if (msgrcv( msqid, &data, sizeof( t_data) - sizeof( long), 0, 0)==-1)
// 		{
// 			perror( "msgrcv() failed");
// 			exit( 1);
// 		}
// 		printf("*** New message received ***\nRaw data: ");
// 		int i;
// 		for(i = 0; i<BUFF_SIZE; i++)
// 			printf("%02X ", data.data_buff[i]);
// 		printf("\n");

// 		if (data.data_type == TYPE_STRING)
// 		{
// 			printf("Interpreted as string: %15s\n", data.data_buff);
// 		}
// 		else if (data.data_type == TYPE_INT_STRING)
// 		{
// 			// memcpy(&dValue1, data.data_buff, sizeof(int));
// 			printf("Interpreted as int + string: %d, %15s\n", data.data_num, data.data_buff);
// 		}
// 		else if (data.data_type == TYPE_TWODOUBLES)
// 		{
// 			memcpy(&dValue1, data.data_buff, sizeof(double));
// 			memcpy(&dValue2, data.data_buff+sizeof(double), sizeof(double));
// 			printf("Interpreted as two doubles: %f, %f\n", dValue1, dValue2);
// 		}
// 		else if (data.data_type == TYPE_ARRAY)
// 		{
// 			memcpy(new_array, data.data_buff, BUFF_SIZE);
// 			printf("Interpreted as array: ");
// 			int i;
// 			for(i = 0; i<BUFF_SIZE; i++)
// 				printf("%d ", new_array[i]);
// 			printf("\n");
// 		}
// 		else if (data.data_type == TYPE_DOUBLEANDARRAY)
// 		{
// 			memcpy(&dValue1, data.data_buff, sizeof(double));
// 			memcpy(new_array, data.data_buff+sizeof(double), BUFF_SIZE/2);
// 			printf("Interpreted as one double and array: %f, ", dValue1);
// 			int i;
// 			for(i = 0; i<BUFF_SIZE/2; i++)
// 				printf("%d ", new_array[i]);
// 			printf("\n");
// 		}

// 	}
// }
