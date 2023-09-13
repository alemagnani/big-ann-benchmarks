To run locally without the container you can use the `build.local.sh` to build the library locally
Then you can use:

`PYTHONPATH=/home/alessandro/workspace/faiss/build/faiss/python/build/lib/:/home/alessandro/workspace/big-ann-benchmarks/neurips23/filter/wm_filter/ python run.py --neurips23track filter    --algorithm wm_filter   --dataset yfcc-10M --nodocker`

The `PYTHONPATH` has to include both the Faiss library path as well the as the `wm_filter` folder where the library was build by `build_local.sh`

With a frequency threshold if we run only queries with a lower frequency we have

| Threshold | % run | estimated products |
|-----------|-----| ------|
| 0.00001   | 20%| 100 |
| 0.0001    | 34% | 1000 |
| 0.001     | 49% | 10000 |
| 0.01      | 70% | 100000|
| 0.1       | 94% | 1000000 |

We are goingi to use these buckets to find the best tradeoff between max_codes and probes
Rougly the lower the frequency the faster it is to get results with high recall. 
So the optimum point will be one with higher recall for lower frequency and lower recall for higher frequencies

|Threshold | %run | Name     |
|-----| -----|----------|
| 0.00001   | 20% | Bucker 1 |
| 0.00026   | 40%| Bucket 2 |
| 0.0037 | 60% | Bucket 3 |
|0.025 | 80% | Bucket 4 |
|   | Above | Bucket 5 |
