# Data Science on Cloud Foundry

This is a full end-to-end data science example running on Cloud Foundry (CF).

## Getting Started

#### Concourse CI
1. Clone the repo: `git clone https://github.com/datitran/cf-demo.git`
2. Change to the cloned repo and do: `vagrant up`
3. Target your local VirtualBox: `fly -t example-ci login -c http://192.168.100.4:8080`
4. Run the pipeline: `fly -t example-ci set-pipeline -p data-science-ci -c pipeline.yml -l credentials.yml`
5. Unpause the pipeline: `fly -t example-ci unpause-pipeline -p data-science-ci`

#### Docker
1. Build docker image from file: `docker build -f dockerfile -t datitran/cf-demo .`

## Dependencies
- [Anaconda](https://www.continuum.io/downloads) Python 3.5.2
- Python conda environment (install with `conda env create --file environment.yml`)
- [PCF Dev](https://github.com/pivotal-cf/pcfdev)
- [Concourse](http://concourse.ci/index.html)
- [Docker](https://www.docker.com/)
- [Vagrant](https://www.vagrantup.com/)

## Copyright

See [LICENSE](LICENSE) for details.
Copyright (c) 2016 [Dat Tran](http://www.dat-tran.com/).
