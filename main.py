import train
import test

print("Training started...")
train.train()

test.print_filters()

print("Testing started...")
test.test()