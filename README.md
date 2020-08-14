# tensor-coin
Blockchain, polymorphic (runtime CPU/GPU dynamic type) uhash, and validator, 
built from scratch.
Note that the tensorhash algorithm is only conceptual - it is an example for 
matrix multiplication intensive hash, and is not optimal
(TODO:  
CPU/GPU multithreading, sha256 on the GPU only without re-uploading data,
maybe make the algorithm more AES-like, the whole coin transactions isn't 
implemented - specify how should a transaction look like, save them in a 
Merkle tree.
Add specification of chain/block data structures so they can be saved as a JSON
and sent over network to the verifier (auth_wallet).
)

