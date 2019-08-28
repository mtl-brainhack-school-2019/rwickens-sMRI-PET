#written by atrophiedbrain - Joshua Morse

import numpy as np
import pickle
import argparse

def get_age_index(age, times):
    return np.where(times == age)

def random_ct(count=1):
    set = np.arange(2.7, 3.98, 0.01, dtype=float)
    return np.random.choice(set, size=count)


def random_age(min, max, count=1):
    set = np.arange(min, max, 0.5, dtype=float)
    return np.random.choice(set, size=count)


def random_ctp_ad(count=1):
    set = np.arange(-0.05, -0.005, 0.005, dtype=float)
    set2 = np.arange(0.005, 0.005, 0.005, dtype=float)
    set = np.concatenate((set, set2))
    return np.random.choice(set, size=count)


def random_ctp_hc(count=1):
    set = np.arange(-0.03, -0.005, 0.005, dtype=float)
    set2 = np.arange(0.005, 0.005, 0.005, dtype=float)
    set = np.concatenate((set, set2))
    return np.random.choice(set, size=count)


def estimate(p, times, ivs, step):
    output = np.zeros((times.shape[0], ivs.shape[0]))
    output[0, :] = ivs
    i = 1
    for time in times[1:]:
        last = output[i - 1, :]

        deriv = last*p
        output[i, :] = last + deriv * step
        i += 1

    return output


parser = argparse.ArgumentParser('simulated data generator')
parser.add_argument('--num_subjects', type=int, default=1000)
parser.add_argument('--num_param_groups', type=int, default=10)
parser.add_argument('--num_regions', type=int, default=64)
parser.add_argument('--min_age', type=float, default=60.0)
parser.add_argument('--max_age', type=float, default=80.0)
parser.add_argument('--num_timepoints', type=int, default=3)
parser.add_argument('--num_severe_regions', type=int, default=6)
parser.add_argument('--severity_multiplier', type=float, default=1.5)
args = parser.parse_args()

ctp_ad = np.zeros((args.num_subjects, args.num_regions), dtype=float)
ctp_hc = np.zeros((args.num_subjects, args.num_regions), dtype=float)

# Generate simulated parameters
for i in np.arange(0, args.num_subjects):
    ctp_ad[i, :] = random_ctp_ad(args.num_regions)
    ctp_hc[i, :] = random_ctp_hc(args.num_regions)

# Increase atrophy rate in severe regions, randomly selected
# This will create features that can more easily differentiate between AD/HC
if args.num_severe_regions > 0:
    rgs = np.random.choice(np.arange(0, args.num_regions), args.num_severe_regions, replace=False)
    ctp_ad[:, rgs] *= args.severity_multiplier

TIME_STEP = 0.5
TIMES = np.arange(args.min_age, args.max_age, TIME_STEP)

# Allocate arrays for all data
ad_ages = np.zeros(args.num_subjects)
hc_ages = np.zeros(args.num_subjects)

ad_ct_cs = np.zeros((args.num_subjects, args.num_regions))
hc_ct_cs = np.zeros((args.num_subjects, args.num_regions))

ad_ct_long = np.zeros((args.num_subjects, 1+args.num_timepoints, args.num_regions))
hc_ct_long = np.zeros((args.num_subjects, 1+args.num_timepoints, args.num_regions))

# Generate simulated trajectories from simulated parameters
for i in np.arange(0, args.num_subjects):
    # Get a unique starting point for each subject
    ct0_ad = random_ct(args.num_regions)
    ct0_hc = random_ct(args.num_regions)

    # Generate a random age for each subject (and make sure we have enough timepoints for longitudinal data)
    age_ad = random_age(args.min_age, args.max_age-args.num_timepoints, 1)
    age_hc = random_age(args.min_age, args.max_age-args.num_timepoints, 1)

    age_ad_ti = get_age_index(age_ad, TIMES)
    age_hc_ti = get_age_index(age_hc, TIMES)

    # Simulate trajectories from the simulated parameters
    estims_ad = estimate(ctp_ad[i, :], TIMES, ct0_ad, TIME_STEP)
    estims_hc = estimate(ctp_hc[i, :], TIMES, ct0_hc, TIME_STEP)

    # The estims_* arrays now have values across the age range for a subject

    # Use estimated trajectories to save cross-sectional data (one age)
    ad_ages[i] = age_ad
    ad_ct_cs[i, :] = estims_ad[age_ad_ti]
    hc_ages[i] = age_hc
    hc_ct_cs[i, :] = estims_hc[age_hc_ti]

    # Use estimated trajectories to save longitudinal data
    ad_ct_long[i, 1, :] = estims_ad[age_ad_ti]
    hc_ct_long[i, 1, :] = estims_hc[age_hc_ti]
    for j in np.arange(1, args.num_timepoints):
        ad_ct_long[i, 1+j, :] = estims_ad[age_ad_ti+j]
        hc_ct_long[i, 1+j, :] = estims_hc[age_hc_ti+j]

# Pickle data (save Python objects to disk)
with open("data.pkl", "ab") as f:
    pickle.dump(ad_ages, f)
    pickle.dump(ad_ct_cs, f)
    pickle.dump(ad_ct_long, f)
    pickle.dump(hc_ages, f)
    pickle.dump(hc_ct_cs, f)
    pickle.dump(hc_ct_long, f)
