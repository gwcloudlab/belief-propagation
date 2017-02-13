################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/bnf-parser/expression.c \
../src/bnf-parser/main.c 

OBJS += \
./src/bnf-parser/expression.o \
./src/bnf-parser/main.o 

C_DEPS += \
./src/bnf-parser/expression.d \
./src/bnf-parser/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/bnf-parser/%.o: ../src/bnf-parser/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/bnf-parser" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


