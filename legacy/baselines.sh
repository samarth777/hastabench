set -x
for m in noop ik_oracle; do scripts/cluster/job.sh -m hastabench.baselines --method $m --budget 1; done
for b in 40 400 2000; do for m in random_image es_image; do scripts/cluster/job.sh -m hastabench.baselines --method $m --budget $b; done; done
