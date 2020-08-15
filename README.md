# tensor-coin
Blockchain, polymorphic (runtime CPU/GPU dynamic type) uhash, and validator, 
built from scratch.
Note that the tensorhash algorithm is only conceptual - it is an example for 
matrix multiplication intensive hash, and is not optimal. The current GPU code
isn't parallel and only partial GPU which makes it very slow (many uploads...).
TODO:  
CPU/GPU multithreading, sha256 on the GPU only without re-uploading data
Then, increase MAT_MULT_ITERS.
Coin transactions aren't currently checked - 
specify how should a transaction look like, save them as a Merkle tree, and 
add verification functions.
Add specification of chain/block data structures so they can be serialized as a
JSON and sent over network to the verifier (auth_wallet).
