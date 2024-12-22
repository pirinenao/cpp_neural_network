# Compiler
COMPILER = g++

# Compiler flags
FLAGS = -std=c++11 -Wall -Wextra -Iinclude -DMNIST_DATA_LOCATION=\"$(MNIST_DATA_DIR)\"

# Target executable
TARGET = ./main

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# MNIST loader include directory
MNIST_INCLUDE_DIR = ./include/mnist
MNIST_DATA_DIR = ./include/mnist/datasets

# Source and object files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Default target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(COMPILER) $(FLAGS) -I$(MNIST_INCLUDE_DIR) -o $@ $^

# Rule to build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(COMPILER) $(FLAGS) -I$(MNIST_INCLUDE_DIR) -c $< -o $@

# Ensure the build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build files
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean
