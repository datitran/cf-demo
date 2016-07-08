#!/bin/sh

set -e

cd data-science-ci
nosetests -vs src/python/tests

