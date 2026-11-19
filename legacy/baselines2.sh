scripts/cluster/job.sh -m hastabench.baselines --method noop --budget 1 --targets samyuta --seeds 0 1
for b in 60 600; do for m in random_image es_image; do scripts/cluster/job.sh -m hastabench.baselines --method $m --budget $b --targets samyuta --seeds 0 1; done; done
