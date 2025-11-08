# Makefile for dkm benchmark and tests

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -I./include -march=native -mavx2
LDFLAGS := -lpthread

# OpenMP flag
OMP_FLAG := -fopenmp

# OpenCV libraries using pkg-config
# This makes the Makefile more portable. 2>/dev/null suppresses errors if not found.
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || echo "")
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null || echo "")

# Directories
BENCH_SRC_DIR := src/bench
TEST_SRC_DIR := src/test
BUILD_DIR := build
BENCH_TARGET_DIR := $(BUILD_DIR)/bench
TEST_TARGET_DIR := $(BUILD_DIR)/test

# Target executables (internal build targets)
BENCH_BUILD_TARGET := $(BENCH_TARGET_DIR)/bench
TEST_BUILD_TARGET := $(TEST_TARGET_DIR)/test

# Source files
BENCH_SOURCES := $(BENCH_SRC_DIR)/bench.cpp
TEST_SOURCES := $(TEST_SRC_DIR)/test.cpp

# Object files directories
BENCH_OBJ_DIR := $(BENCH_TARGET_DIR)/obj
TEST_OBJ_DIR := $(TEST_TARGET_DIR)/obj

# Generate object file names from source files
BENCH_OBJECTS := $(patsubst $(BENCH_SRC_DIR)/%.cpp, $(BENCH_OBJ_DIR)/%.o, $(BENCH_SOURCES))
TEST_OBJECTS := $(patsubst $(TEST_SRC_DIR)/%.cpp, $(TEST_OBJ_DIR)/%.o, $(TEST_SOURCES))

# Phony targets (targets that are not files)
.PHONY: all bench test run clean help build

# Default target: build both bench and test executables
all: build

# Alias for the 'all' target for clarity
build: $(BENCH_BUILD_TARGET) $(TEST_BUILD_TARGET)

# === Benchmark Targets ===
# 'make bench' will now compile and run the benchmark
bench: $(BENCH_BUILD_TARGET)
	@echo "\n=== Running Benchmark ==="
	@cp $(BENCH_SRC_DIR)/*.csv $(BENCH_TARGET_DIR)/ 2>/dev/null || :
	@cd $(BENCH_TARGET_DIR) && ./$(notdir $(BENCH_BUILD_TARGET))

# Rule to link the benchmark executable
$(BENCH_BUILD_TARGET): $(BENCH_OBJECTS)
	@echo "Linking benchmark executable: $@"
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OMP_FLAG) $^ -o $@ $(LDFLAGS) $(OPENCV_LIBS)

# Rule to compile benchmark source
$(BENCH_OBJ_DIR)/%.o: $(BENCH_SRC_DIR)/%.cpp
	@echo "Compiling benchmark source: $<"
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OMP_FLAG) $(OPENCV_CFLAGS) -MMD -MP -c $< -o $@


# === Test Targets ===
# 'make test' will now compile and run the tests
test: $(TEST_BUILD_TARGET)
	@echo "\n=== Running Tests ==="
	@cp $(TEST_SRC_DIR)/*.csv $(TEST_TARGET_DIR)/ 2>/dev/null || :
# 	@cp $(BENCH_SRC_DIR)/*.csv $(TEST_TARGET_DIR)/ 2>/dev/null || :
	@cd $(TEST_TARGET_DIR) && ./$(notdir $(TEST_BUILD_TARGET))

# Rule to link the test executable
$(TEST_BUILD_TARGET): $(TEST_OBJECTS)
	@echo "Linking test executable: $@"
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OMP_FLAG) $^ -o $@ $(LDFLAGS) $(OPENCV_LIBS)

# Rule to compile test source
$(TEST_OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cpp
	@echo "Compiling test source: $<"
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OMP_FLAG) $(OPENCV_CFLAGS) -I$(TEST_SRC_DIR) -MMD -MP -c $< -o $@


# Include the generated dependency files
-include $(BENCH_OBJECTS:.o=.d)
-include $(TEST_OBJECTS:.o=.d)


# === Combined & Utility Targets ===
# 'make run' will execute both bench and test targets
run: bench test
	@echo "\n=== Finished running all targets ==="

clean:
	@echo "Cleaning up build files..."
	@rm -rf $(BUILD_DIR)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Primary Run Targets:"
	@echo "  bench      Compile and run the benchmark."
	@echo "  test       Compile and run the tests."
	@echo "  run        Compile and run both the benchmark and the tests."
	@echo ""
	@echo "Build-only Targets:"
	@echo "  all        (default) Compile both executables without running."
	@echo "  build      Alias for 'all'."
	@echo ""
	@echo "Utility Targets:"
	@echo "  clean      Remove all build files."
	@echo "  help       Show this help message."