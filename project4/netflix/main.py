import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K=[1,2,3,4]
for k in K:
    min_cost=float('inf')
    for seed in [0,1,2,3,4]:
        mixture,post=common.init(X, k, seed)
        cost=kmeans.run(X, mixture, post)
        if cost[2]<min_cost:
            min_cost=cost[2]
            best_seed=seed
            best_mixture=mixture
            best_post=post
    print(f"k={k}, min_cost={min_cost}")
    common.plot(X, best_mixture, best_post, f"k{k} best seed{best_seed}, min cost{min_cost}")
