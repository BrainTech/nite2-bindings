# location of the Python header files
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# location of the Boost Python include files and library
NITE2_INC = /home/alex/NiTE-Linux-x64-2.2/Include
OPENNI2_INC = /home/alex/OpenNI2/Include
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib

CC = g++

# compile mesh classes
TARGET = nite2

OBJS = nite2.o openni2.o wrapper.o

$(TARGET).so: $(OBJS)
	$(CC) -shared -Wl,--export-dynamic $^ -L$(BOOST_LIB) -lboost_python-$(PYTHON_VERSION) -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $@

$(OBJS): %.o: %.cpp
	$(CC) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -I$(NITE2_INC) -I$(OPENNI2_INC) -std=gnu++11 -fPIC -c $< -o $@

