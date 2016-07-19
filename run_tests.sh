#!/bin/sh

set -e

cd cf-demo-testing
nosetests -vs src/python/tests

