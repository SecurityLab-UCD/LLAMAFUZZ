////////////////////////// main_sender.c
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>


#define  BUFF_SIZE   16 

int TYPE_REQUEST = 1;
int TYPE_REWARD = 2;
int TYPE_SEED = 3;
int TYPE_EMPTY_SEED = 4;

typedef struct {
  long data_type;
  char data_buff[BUFF_SIZE];
} My_message;

int main( void)
{
    /* the mutation, send request to LLM, then receive mutate seed */
  //   u32 out_buf_len = afl_mutate(data->afl, data->buf, buf_size,
  //   havoc_steps,false, true, add_buf, add_buf_size, max_size);
  My_message my_msg;
  int        msg = 200;
  int        msqid;
  int        receive_flag = 1;

  // Create or open the message queue
  if ((msqid = msgget((key_t)1234, IPC_CREAT | 0666)) == -1) {
    perror("msgget() failed");
    exit(1);
  }
// send the request message
  memcpy(my_msg.data_buff, &msg, sizeof(int));
  my_msg.data_type = TYPE_REQUEST;
  if (msgsnd(msqid, &my_msg, sizeof(my_msg.data_buff), 0) == -1) {
    perror("msgsnd() failed");
    exit(1);
  }
  // receive seed info from llm
  clock_t start_time;
  start_time = clock();

  while (1) {
    // if run time exceed 0.1s then break and mutate default one
    if (difftime(clock(), start_time) >= 0.1) {
      receive_flag = 0;
      break;
    }
    
    if (-1 == msgrcv(msqid, &my_msg, sizeof(My_message) - sizeof(long), 0, 0)) {
      perror("msgrcv() failed");
      exit(1);
    } 
    else {
      if (my_msg.data_type == TYPE_SEED){
        receive_flag = 1;
        printf("Interpreted as string: %15s\n", my_msg.data_buff);
      }
      else{
        // receive empty seed
        receive_flag = 0;
      }
      break;
    }
  }
}