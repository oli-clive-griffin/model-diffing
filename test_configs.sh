#!/bin/bash

# Set timeout duration in seconds
# TIMEOUT=200

# Function to test a run.py with its config files
test_experiment() {
    exp_dir="model_diffing/scripts/$1"
    echo "Testing $exp_dir..."
    
    # Find all yaml files in the experiment directory
    for config in "$exp_dir"/*.yaml; do
        if [ -f "$config" ]; then
            echo "  Testing with config: $config"
            # timeout $TIMEOUT 
            python "$exp_dir/run.py" "$config"
            
            # Check exit status
            status=$?
            if [ $status -eq 124 ]; then
                echo "  ✓ Started successfully (timed out as expected)"
            elif [ $status -eq 0 ]; then
                echo "  ✓ Completed successfully"
            else
                echo "  ✗ Failed with exit code $status"
            fi
            echo
        fi
    done
}

# Test each experiment
test_experiment "skip_trans_crosscoder"
test_experiment "train_jan_update_crosscoder"
test_experiment "train_jumprelu_sliding_window" 
test_experiment "train_l1_crosscoder"
test_experiment "train_l1_sliding_window"
test_experiment "train_topk_crosscoder"