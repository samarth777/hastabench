for m in rest mean_shape ik_oracle cma_score cma_image; do scripts/cluster/job.sh -m hastabench.refbase --method $m; done
for m in rest ik_oracle cma_score cma_image; do scripts/cluster/job.sh -m hastabench.refbase --method $m --targets samyuta; done
