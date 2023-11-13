CC=gcc
OBJ=main_sender.o main_receiver.o sender.o
TARGET=sender receiver sdr


%.o: %.c
	$(CC) -c -o $@ $<

all: $(OBJ)
	$(CC) -o sender main_sender.o
	$(CC) -o receiver main_receiver.o
	$(CC) -o sdr sender.o

clean:
	rm -f $(OBJ) $(TARGET)
