import train
import test

print("Training started...")
train.train()

test.print_sample_filters()

print("Testing started...")
test.test_machine()