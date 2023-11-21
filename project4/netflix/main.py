import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here


def run_naive_em(X, K, seed):
    mixture, post = common.init(X, K, seed)
    mixture, post, log_likelihood = naive_em.run(X, mixture, post)
    return mixture, post, log_likelihood



if __name__ == "__main__":

    run_kmeans_fcn = False
    run_naive_em_fcn = True

    if run_naive_em_fcn:
        X = np.loadtxt("toy_data.txt")
        K = 3
        final_mixture, final_post, log_likelihood = run_naive_em(X, K, 0)
        print("Final log likelihood:", log_likelihood)
        common.plot(X, final_mixture, final_post, "Final Clustering")

    if run_kmeans_fcn:
        X = np.loadtxt("toy_data.txt")
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
