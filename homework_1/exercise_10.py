import numpy as np
import matplotlib.pyplot as plt


def generate_bernoulli_samples():
    samples = []
    for i in range(0, 1000):
        bernoulli_samples = np.random.binomial(1, 0.75, 100)
        mean = (np.sum(bernoulli_samples) - 75)/(10*np.sqrt(0.1875))
        samples.append(mean)
    plt.hist(samples,
             bins=10,
             rwidth=0.8, cumulative=True, density=True
             )
    plt.xlabel("X")
    plt.ylabel("Probability")
    # plt.xticks([ 0.4,0.5,0.6,0.7, 0.8, 0.9,1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('10/bernoulli_cdf_histogram.png')
    #plt.show()
    plt.clf()
    plt.scatter(range(1000), samples, s=10, alpha=0.7, edgecolors='none')
    plt.ylim(-30, 30)
    # plt.yticks([0, 0.25,0.5,0.75,1])
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('10/bernoulli_time_series.png')
    plt.show()
    print(f'expected value is {np.mean(samples)}')
    print(f'standard deviation: {np.var(samples)}')


def generate_uniform_samples():
    samples = []
    for i in range(0, 1000):
        uniform_samples = np.random.uniform(0, 1, 100)
        mean = (np.sum(uniform_samples) - 50) / (10 * np.sqrt(0.0833))
        samples.append(mean)
    plt.hist(samples,
             bins=20,
             rwidth=0.8, cumulative=True, density=True
             )
    plt.xlabel("X")
    plt.ylabel("Probability")
    # plt.xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('10/uniform_cdf_histogram.png')
    #plt.show()
    plt.clf()
    plt.scatter(range(1000), samples, s=10, alpha=0.7, edgecolors='none')
    plt.ylim(-30, 30)
    # plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('10/uniform_time_series.png')
    plt.show()
    print(f'expected value is {np.mean(samples)}')
    print(f'standard deviation: {np.var(samples)}')


def generate_binomial_samples():
    samples = []
    for i in range(0, 1000):
        binomial_samples = np.random.binomial(10, 0.75, 100)
        mean = (np.sum(binomial_samples) - 750) / (10 * np.sqrt(1.875))
        samples.append(mean)
    plt.hist(samples,
             bins=20,
             rwidth=0.8, cumulative=True, density=True
             )
    # plt.xticks([7, 7.1, 7.2, 7.3, 7.5, 7.6, 7.7, 7.8, 7.9, 8])
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('10/binomial_cdf_histogram.png')
    #plt.show()
    plt.clf()
    plt.scatter(range(1000), samples, s=10, alpha=0.7, edgecolors='none')
    plt.ylim(-30, 30)
    #plt.yticks([7.0, 7.2, 7.4, 7.6, 7.8, 8])
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('10/binomial_time_series.png')
    plt.show()
    print(f'expected value is {np.mean(samples)}')
    print(f'standard deviation: {np.var(samples)}')


def generate_gaussian_samples():
    samples = []
    for i in range(0, 1000):
        gaussian_samples = np.random.normal(1, np.sqrt(5), 100)
        mean = (np.sum(gaussian_samples) - 100) / (10 * np.sqrt(5))
        samples.append(mean)
    plt.hist(samples,
             bins=20,
             rwidth=0.8, cumulative=True, density=True
             )
    # plt.xlim(0, 2)
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('10/gaussian_cdf_histogram.png')
    #plt.show()
    plt.clf()
    plt.scatter(range(1000), samples, s=10, alpha=0.7, edgecolors='none')
    plt.ylim(-30, 30)
    # plt.yticks([7.0, 7.2, 7.4, 7.6, 7.8, 8])
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('10/gaussian_time_series.png')
    plt.show()
    print(f'expected value is {np.mean(samples)}')
    print(f'standard deviation: {np.var(samples)}')


def generate_beta_samples(a, b):
    samples = []
    for i in range(0, 1000):
        beta_samples = np.random.beta(a, b, 100)
        mean = (np.sum(beta_samples) - 50) / (10 * np.sqrt(0.0833))
        samples.append(mean)
    plt.hist(samples,
             bins=20,
             rwidth=0.8, cumulative=True, density=True
             )
    #plt.xlim(0.4, 0.6)
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'10/beta_cdf_histogram_b={b}.png')
    #plt.show()
    plt.clf()
    plt.scatter(range(1000), samples, s=10, alpha=0.7, edgecolors='none')
    plt.ylim(-30, 30)
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'10/beta_time_series_b={b}.png')
    plt.show()
    print(f'expected value is {np.mean(samples)}')
    print(f'standard deviation: {np.var(samples)}')


def main():
    generate_bernoulli_samples()
    # generate_uniform_samples()
    # generate_binomial_samples()
    # generate_gaussian_samples()
    # generate_beta_samples(1, 1)



if __name__ == '__main__':
    main()