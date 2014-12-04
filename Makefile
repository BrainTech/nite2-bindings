# location of the Python header files
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# location of the Boost Python include files and library
OPENNI2_INC = ../kinect/Include
OPENNI2_LIB = ../kinect/Redist
NITE2_INC = ../kinect/Include
NITE2_LIB = ../kinect/Redist
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib

CC = g++

TARGET = nite2
OBJS = nite2.o openni2.o wrapper.o

$(TARGET).so: $(OBJS)
	$(CC) -shared -Wl,--export-dynamic $^ -L$(BOOST_LIB) -L$(OPENNI2_LIB) -L$(NITE2_LIB) -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -lboost_python-py27 -lOpenNI2 -lNiTE2 -o $@

$(OBJS): %.o: %.cpp
	$(CC) -Wall -Wextra -O2 -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -I$(NITE2_INC) -I$(OPENNI2_INC) -std=gnu++11 -fPIC -c $< -o $@

clean:
	rm -f *.o
	rm -f *.so

.PHONY: clean

