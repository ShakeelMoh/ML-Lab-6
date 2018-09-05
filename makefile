#My own makefile for ML lab 6

#Declare variables
CC=g++
LIBS=-lm

#First create ".exe" called backpropagation
backpropagation: backpropagation.o
	$(CC) backpropagation.o -o backpropagation $(LIBS)

#Need to make backpropagation.o file though
backpropagation.o: backpropagation.cpp
	$(CC) -c backpropagation.cpp


#Other rules

#Clean .o and exe
clean:
	@rm -f *.o
	@rm -f backpropagation

#To run program
run:
	./backpropagation
