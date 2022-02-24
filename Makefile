default:
	echo "run make specifying a benchmark"

cifar10:
	mkdir -p cifar10/log
	mkdir -p cifar10/saved_models

mnist:
	mkdir -p mnist/log
	mkdir -p mnist/saved_models

clean:
	rm -rf data
	rm -rf __pycache__
	rm -rf cifar10/__pycache__
	rm -rf mnist/__pycache__
	rm -rf models/__pycache__