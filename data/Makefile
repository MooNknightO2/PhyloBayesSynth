CXX=g++
CPPFLAGS=-std=c++11 -I"C:\Program Files\ncl\include"
LDFLAGS=-L"C:\Program Files\ncl\lib\ncl"
LDLIBS=-lncl

all: nexus2phylojson

nexus2phylojson: nexus2phylojson.o
	$(CXX) -o nexus2phylojson nexus2phylojson.o $(LDLIBS) $(LDFLAGS)

clean:
	rm nexus2phylojson.o