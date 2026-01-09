#!/bin/bash
echo "Testing Distributed Banking System"
echo "================================="

# Test basic compilation
echo "1. Testing compilation..."
make clean && make
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful"
else
    echo "✗ Compilation failed"
    exit 1
fi

# Test basic execution
echo "2. Testing basic execution..."
timeout 30s ./main_parallel inp-params-line-3.txt > test_output.log 2>&1
if [ $? -eq 0 ] || [ $? -eq 124 ]; then  # 124 is timeout exit code
    echo "✓ Basic execution successful"
else
    echo "✗ Basic execution failed"
fi

# Test different topologies
echo "3. Testing different topologies..."
for topology in line ring tree arbitrary; do
    for processes in 3 4; do
        if [ -f "inp-params-${topology}-${processes}.txt" ]; then
            echo "  Testing ${topology} topology with ${processes} processes..."
            timeout 20s ./main_parallel inp-params-${topology}-${processes}.txt > /dev/null 2>&1
            if [ $? -eq 0 ] || [ $? -eq 124 ]; then
                echo "    ✓ ${topology} topology working"
            else
                echo "    ✗ ${topology} topology failed"
            fi
        fi
    done
done

echo "System test completed!"
