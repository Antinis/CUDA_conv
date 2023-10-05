TARGET = conv
SOURCE = conv_im2col.cu

NVCC = nvcc
NVCCFLAGS += -O3 -cudart=shared -Xcompiler -fopenmp --ptxas-options=-v

$(TARGET):$(SOURCE)
	$(NVCC) $(SOURCE) -o $(TARGET) $(NVCCFLAGS)

.PHONY:clean
clean:
	rm -rf $(TARGET)