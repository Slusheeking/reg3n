# Makefile for HFT Kernel Module
obj-m += hft_kernel_module.o

# Kernel build directory
KERNEL_DIR := /lib/modules/$(shell uname -r)/build

# Current directory
PWD := $(shell pwd)

# Default target
all:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

# Clean target
clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean
	rm -f *.symvers *.order *.markers

# Install target (requires root)
install: all
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules_install
	depmod -a

# Load module (requires root)
load:
	insmod hft_kernel_module.ko

# Unload module (requires root)
unload:
	rmmod hft_kernel_module

# Show module info
info:
	modinfo hft_kernel_module.ko

# Show statistics
stats:
	cat /proc/hft_stats

# Help target
help:
	@echo "HFT Kernel Module Build System"
	@echo "Available targets:"
	@echo "  all     - Build the kernel module"
	@echo "  clean   - Clean build artifacts"
	@echo "  install - Install module to system (requires root)"
	@echo "  load    - Load module into kernel (requires root)"
	@echo "  unload  - Unload module from kernel (requires root)"
	@echo "  info    - Show module information"
	@echo "  stats   - Show HFT processing statistics"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Linux kernel headers: apt-get install linux-headers-$(shell uname -r)"
	@echo "  - Build essentials: apt-get install build-essential"
	@echo "  - Root privileges for install/load/unload operations"

.PHONY: all clean install load unload info stats help
