opencv_header=-I/usr/include/
opencv_library=-L/usr/local/lib
CXX_FLAGS=-D_MWAITXINTRIN_H_INCLUDED --std=c++11

all: solution

solution: solution.cu
	nvcc solution.cu $(opencv_header) $(opencv_library) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs $(CXX_FLAGS) -arch sm_20 -o solution 

clean:
	rm -rf solution
