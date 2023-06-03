all: hough

hough: houghBaseGlobal.o pgm.o
	nvcc houghBaseGlobal.o pgm.o -o hough

houghBaseGlobal.o: houghBaseGlobal.cu
	nvcc -c houghBaseGlobal.cu -o houghBaseGlobal.o

pgm.o: pgm.cpp
	nvcc -c pgm.cpp -o pgm.o
