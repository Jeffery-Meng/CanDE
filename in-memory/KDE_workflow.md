1. Create a json config file, create result directory, and set all paths.
2. Run FALCONN/src/kde_scripts/auto_prepare.py to automaticall fill data statistics in config, and compute ground truth distances.
3. Use FALCONN/src/kde_scripts/command_console.py to load "FALCONN/src/kde_scripts/auto_param.py parameters"
 (remember to change the json path in auto_param.py) to run grid-search parameter tuning.
4. Pick LSH parameters into config file, change query type to "knn candidates", run "FALCONN/build/falconn -cf [config file]" to print candidates.
5. Change query mode to "hashed queries" and run "FALCONN/build/falconn" again, to print out hashed queries.
6. Change query mode to "probing sequence" and run "FALCONN/build/falconn" again, to print out the probing sequence.
7. Specify bandwidths (gamma), and recall_p file path. Run "FALCONN/build/candidate_probs [config]" to compute the probability (sampling weight) of candidates
8. Run kde_gt to get ground truth kernel densities
9. Run falconn_kde to get kde results and relative error

> New workflow
1. Automatic parameter tuning
python3 FALCONN/src/kde_scripts/falconn_param_tuning.py mnist /media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/ /media/mydrive/KroneckerRotation/data/mnist-train.fvecs /media/mydrive/KroneckerRotation/data/mnist-test.fvecs --iskde --targetrecall 0.6 
2. Print candidates, query hashes, probing sequence and compute pointwise recall value
python3 FALCONN/src/kde_scripts/falconn_candidate_ready.py /media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/mnist
3. Run KDE
python3 FALCONN/src/kde_scripts/falconn_run_kde.py /media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/mnist

> Workflow V3
1. Automatic parameter tuning
python3 FALCONN/src/kde_scripts/falconn_param_tuning.py mnist /media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/ /media/mydrive/KroneckerRotation/data/mnist-train.fvecs /media/mydrive/KroneckerRotation/data/mnist-test.fvecs --targetrecall 0.6 
2. Print precomputed multi-probe sequence, and compute unconditional multi-probe collision probablities via Monte Carlo simulation by ./mp_probs
Also, print kNN candidates.

