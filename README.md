# RQGA DDP


## Description

TBD

## Installation

Clone or download this repository and change directory to `environment`.

### Using Anaconda/Miniconda or Venv

1. Install miniconda
2. Create the environment `conda env create -f environment.yml`
3. Activate the newly created environment
4. Install the necessary libraries: `pip install -r requirements.txt`

### Using Docker

1. Install Docker
2. Run `docker build -t quantum .`
3. Run `docker run -p 8888:8888 -it -v <path-to-workspace>:/app --entrypoint /bin/bash quantum`


## Design

For A=[7,8,18]

we have 3!=6 permutations => 3 qubits to encode.
We can encode each permutation with a 3bits binary value:

Binary	Decimal	Permutation (indices)	Permuted A
000	0	(0,1,2)	(7,8,18)
001	1	(0,2,1)	(7,18,8)
010	2	(1,0,2)	(8,7,18)
011	3	(1,2,0)	(8,18,7)
100	4	(2,0,1)	(18,7,8)
101	5	(2,1,0)	(18,8,7)
110	6	invalid	
111	7	invalid

The same, applied for B, 8! = a lot and it would need 16 or more qubits.
So a table with 8! entries.

Or better, check the ranking/unranking algorithm
