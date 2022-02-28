# Makefile

SHELL := /bin/bash

default:
	echo "run make specifying a benchmark"

cifar10-init:
	mkdir -p cifar10/log
	mkdir -p cifar10/saved_models

kws-init:
	mkdir -p kws/log
	mkdir -p kws/saved_models

mnist-init:
	mkdir -p mnist/log
	mkdir -p mnist/saved_models

vww-init:
	mkdir -p vww/log
	mkdir -p vww/saved_models
	mkdir -p vww/data
	wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz -P vww/data
	tar -xvf vww/data/vw_coco2014_96.tar.gz -C vww/data
	rm vww/data/vw_coco2014_96.tar.gz
	

clean:
	rm -rf data
	rm -rf __pycache__
	rm -rf cifar10/__pycache__
	rm -rf mnist/__pycache__
	rm -rf models/__pycache__
	rm -rf vww/__pycache__