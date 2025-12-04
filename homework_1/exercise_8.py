import numpy as np
import matplotlib.pyplot as plt


def generate_bernoulli_samples(p, n_samples):
    bernoulli_samples = np.random.binomial(1, p, n_samples)
    plt.hist(bernoulli_samples,
             bins=[-0.5, 0.5, 1.5],
             rwidth=0.8, cumulative=True, density=True
             )
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.xticks([0, 1])
    plt.grid(True)
    #plt.savefig('bernoulli_cdf_histogram.png')
    plt.clf()
    # plt.show()
    plt.hist(bernoulli_samples,
             bins=[-0.5, 0.5, 1.5],
             density=True,
             rwidth=0.8)
    plt.xticks([0, 1])
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig('bernoulli_pmf_histogram.png')
    # plt.show()
    plt.clf()
    plt.scatter(range(100), bernoulli_samples[0:100], s=10, alpha=0.7, edgecolors='none')
    plt.ylim(-0.1, 1.2)
    plt.yticks([0, 1])
    plt.xlabel("Time")
    plt.ylabel("x")
    # plt.grid(True, linestyle='--', alpha=0.7)
    #plt.savefig('bernoulli_time_series.png')
    #plt.show()
    # print(f'expected value is {np.sum(bernoulli_samples)/10000}')
    print(f'standard deviation: {np.var(bernoulli_samples)}')


def generate_uniform_samples():
    uniform_samples = np.random.uniform(0, 1, 10000)
    plt.clf()
    plt.hist(uniform_samples,
             bins=20,
             rwidth=0.8, cumulative=True, density=True
             )
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.xticks([0, 1])
    plt.grid(True)
    # plt.savefig('uniform_cdf_histogram.png')
    # plt.show()
    plt.clf()
    plt.hist(uniform_samples,
             bins=30,
             density=True,
             rwidth=0.8)
    plt.xticks([0, 1])
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.savefig('uniform_pdf_histogram.png')
    #plt.show()
    plt.clf()
    plt.scatter(range(100), uniform_samples[0:100], s=10, alpha=0.7, edgecolors='none')
    plt.ylim(-0.1, 1.2)
    plt.yticks([0, 1])
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.savefig('uniform_time_series.png')
    # plt.show()
    print(f'expected value is {np.sum(uniform_samples) / 10000}')
    print(f'standard deviation: {np.var(uniform_samples)}')

def generate_binomial_samples(p, n_samples):
    binomial_samples = np.random.binomial(10, p, n_samples)
    plt.hist(binomial_samples,
             bins=20,
             rwidth=0.8, cumulative=True, density=True
             )
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.xticks([0, 1,2,3,4,5,6,7,8,9,10])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig('binomial_cdf_histogram.png')
    # plt.show()
    plt.clf()
    plt.hist(binomial_samples,
             bins=[-0.5, 0.5, 1.5, 2.5,3.5,4.5,5.5, 6.5,7.5,8.5,9.5, 10.5],
             density=True,
             rwidth=0.8
             )
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig('binomial_pmf_histogram.png')
    # plt.show()
    plt.clf()
    plt.scatter(range(100), binomial_samples[0:100], s=10, alpha=0.7, edgecolors='none')
    plt.ylim(-0.1, 10.5)
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.savefig('binomial_time_series.png')
    # plt.show()
    print(f'expected value is {np.sum(binomial_samples)/10000}')
    print(f'standard deviation: {np.var(binomial_samples)}')


def generate_gaussian_samples():
    gaussian_samples = np.random.normal(1, np.sqrt(5), 10000)
    plt.hist(gaussian_samples,
             bins=20,
             rwidth=0.8, cumulative=True, density=True
             )
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.xticks([-6, -4, -2, 0, 2,4, 6,8, 10])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.savefig('gaussian_cdf_histogram.png')
    #plt.show()
    plt.clf()
    plt.hist(gaussian_samples,
             bins=30,
             density=True,
             rwidth=0.8)
    plt.xticks([-8,-6, -4, -2, 0,1, 2, 4, 6, 8, 10])
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.savefig('gaussian_pdf_histogram.png')
    #plt.show()
    plt.clf()
    plt.scatter(range(100), gaussian_samples[0:100], s=10, alpha=0.7, edgecolors='none')
    # plt.ylim(-0.1, 1.2)
    # plt.yticks([0, 1])
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.savefig('gaussian_time_series.png')
    # plt.show()
    print(f'expected value is {np.sum(gaussian_samples) / 10000}')
    print(f'standard deviation: {np.var(gaussian_samples)}')


def generate_beta_samples(a, b):
    beta_samples = np.random.beta(a, b, 10000)
    plt.hist(beta_samples,
             bins=20,
             rwidth=0.8, cumulative=True, density=True
             )
    plt.xlabel("X")
    plt.ylabel("Probability")
    # plt.xticks([-6, -4, -2, 0, 2, 4, 6, 8, 10])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig(f'beta_cdf_histogram_b={b}.png')
    # plt.show()
    plt.clf()
    plt.hist(beta_samples,
             bins=30,
             density=True,
             rwidth=0.8)
    # plt.xticks([-8, -6, -4, -2, 0, 1, 2, 4, 6, 8, 10])
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig(f'beta_pdf_histogram_b={b}.png')
    # plt.show()
    plt.clf()
    plt.scatter(range(100), beta_samples[0:100], s=10, alpha=0.7, edgecolors='none')
    plt.ylim(0, 1)
    # plt.yticks([0, 1])
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.savefig(f'8/beta_time_series_b={b}.png')
    plt.show()
    print(f'expected value is {np.mean(beta_samples)}')
    print(f'standard deviation: {np.var(beta_samples)}')



def main():
    # generate_bernoulli_samples(0.75, 10000)
    # generate_uniform_samples()
    # generate_binomial_samples(0.75, 10000)
    # generate_gaussian_samples()
    generate_beta_samples(1, 5)



if __name__ == '__main__':
    main()