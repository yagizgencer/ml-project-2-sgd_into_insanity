import nibabel as nib
import numpy as np
from numpy.linalg import norm
from scipy import stats
from math import *
import sys, glob, xlwt, os.path, re
import pandas as pd

ground_truth_path = ""
prediction_path = ""

# read ground-truth
ground_truth = pd.read_pickle(ground_truth_path)

# read predictions
predictions = pd.read_pickle(prediction_path)

# prepare EXCEL output
XLS = xlwt.Workbook()
XLS_sheet = XLS.add_sheet("Local measures")

XLS_sheet.write(0,  0, "filename")

#PERCENTAGE ERROR IN THE ESTIMATION OF NUMBER OF PEAKS
XLS_sheet.write(0,  1, "Pd, mean")
XLS_sheet.write(0,  2, "Pd, std")
XLS_sheet.write(0,  3, "Pd, min")
XLS_sheet.write(0,  4, "Pd, 25 perc")
XLS_sheet.write(0,  5, "Pd, 50 perc")
XLS_sheet.write(0,  6, "Pd, 75 perc")
XLS_sheet.write(0,  7, "Pd, max")

#NUMBER OF UNDERESTIMATION OF NUMBER OF PEAKS
XLS_sheet.write(0,  8, "n-, mean")
XLS_sheet.write(0,  9, "n-, std")
XLS_sheet.write(0, 10, "n-, min")
XLS_sheet.write(0, 11, "n-, 25 perc")
XLS_sheet.write(0, 12, "n-, 50 perc")
XLS_sheet.write(0, 13, "n-, 75 perc")
XLS_sheet.write(0, 14, "n-, max")

#NUMBER OF OVERESTIMATION OF NUMBER OF PEAKS
XLS_sheet.write(0, 15, "n+, mean")
XLS_sheet.write(0, 16, "n+, std")
XLS_sheet.write(0, 17, "n+, min")
XLS_sheet.write(0, 18, "n+, 25 perc")
XLS_sheet.write(0, 19, "n+, 50 perc")
XLS_sheet.write(0, 20, "n+, 75 perc")
XLS_sheet.write(0, 21, "n+, max")

#AVERAGE ANGULAR ERROR
XLS_sheet.write(0, 22, "AE, mean")
XLS_sheet.write(0, 23, "AE, std")
XLS_sheet.write(0, 24, "AE, min")
XLS_sheet.write(0, 25, "AE, 25 perc")
XLS_sheet.write(0, 26, "AE, 50 perc")
XLS_sheet.write(0, 27, "AE, 75 perc")
XLS_sheet.write(0, 28, "AE, max")

#SUM OF FRACTION ERRORS
XLS_sheet.write(0, 29, "FE, mean")
XLS_sheet.write(0, 30, "FE, std")
XLS_sheet.write(0, 31, "FE, min")
XLS_sheet.write(0, 32, "FE, 25 perc")
XLS_sheet.write(0, 33, "FE, 50 perc")
XLS_sheet.write(0, 34, "FE, 75 perc")
XLS_sheet.write(0, 35, "FE, max")


XLS_row = 1

Pd = np.zeros(len(ground_truth))
nP = np.zeros(len(ground_truth))
nM = np.zeros(len(ground_truth))
AE = np.zeros(len(ground_truth))
FE = np.zeros(len(ground_truth))

for i in range(len(ground_truth)):

    DIR_true = np.zeros((3, 5))
    DIR_est = np.zeros((3, 5))

    frac_true = np.zeros(5)
    frac_est = np.zeros(5)

    # compute M_true, DIR_true, M_est, DIR_est
    M_true = 0
    for d in range(5):
        dir = ground_truth[i,range(d*3, d*3+3)]
        f = norm(dir)
        if f > 0:
            frac_true[M_true] = f
            DIR_true[:,M_true] = dir / f
            M_true += 1
    frac_true /= frac_true.sum()

    M_est = 0
    for d in range(5):
        dir = predictions[i,range(d*3, d*3+3)]
        f = norm(dir)
        if f > 0:
            frac_est[M_est] = f
            DIR_est[:,M_est] = dir / f
            M_est += 1
    frac_est /= frac_est.sum()

    # compute Pd, nM and nP
    M_diff = M_true - M_est
    Pd[i] = 100 * abs(M_diff) / M_true
    if M_diff > 0:
        nM[i] = M_diff
    else:
        nP[i] = -M_diff

    # ANGULAR ACCURACY

    # precompute matrix with angular errors among all estimated and true fibers
    A = np.zeros((M_true, M_est))
    for j in range(0,M_true):
        for k in range(0,M_est):
            err = acos(min(1.0, abs(np.dot(DIR_true[:,j], DIR_est[:,k])))) # crop to 1 for internal precision
            A[j,k] = min(err, pi-err) / pi * 180

    # compute the "base" error
    M = min(M_true,M_est)
    err = np.zeros(M)
    frac_err = np.zeros(M)
    notUsed_true = np.array(range(M_true))
    notUsed_est = np.array(range(M_est))
    AA = np.copy(A)
    for j in range(0, M):
        err[j] = np.min(AA)
        r, c = np.nonzero(AA == err[j])
        frac_err[j] = abs(frac_true[r] - frac_est[c])
        AA[r[0],:] = float('Inf')
        AA[:,c[0]] = float('Inf')
        notUsed_true = notUsed_true[notUsed_true != r[0]]
        notUsed_est = notUsed_est[notUsed_est != c[0]]

    # account for OVER-ESTIMATES
    if M_true < M_est:
        if M_true > 0:
            for j in notUsed_est:
                err = np.append(err, min(A[:, j]))
                frac_err = np.append(frac_err, frac_est[j])
        else:
            err = np.append(err, 45)
            frac_err = np.append(frac_err, 1)

    # account for UNDER-ESTIMATES
    elif M_true > M_est:
        if M_est > 0:
            for j in notUsed_true:
                err = np.append(err, min(A[j,:]))
                frac_err = np.append(frac_err, frac_true[j])
        else:
            err = np.append(err, 45)
            frac_err = np.append(frac_err, 1)

    AE[i] = err.mean()
    FE[i] = frac_err.sum()


XLS_sheet.write(XLS_row, 0,  "cv metrics")

# PERCENTAGE ERROR IN THE ESTIMATION OF NUMBER OF PEAKS
XLS_sheet.write(XLS_row,  1, np.mean(Pd))
XLS_sheet.write(XLS_row,  2, np.std(Pd))
XLS_sheet.write(XLS_row,  3, np.min(Pd))
XLS_sheet.write(XLS_row,  4, stats.scoreatpercentile(Pd,25))
XLS_sheet.write(XLS_row,  5, np.median(Pd))
XLS_sheet.write(XLS_row,  6, stats.scoreatpercentile(Pd,75))
XLS_sheet.write(XLS_row,  7, np.max(Pd))

# NUMBER OF UNDERESTIMATION OF NUMBER OF PEAKS
XLS_sheet.write(XLS_row,  8, np.mean(nM))
XLS_sheet.write(XLS_row,  9, np.std(nM))
XLS_sheet.write(XLS_row, 10, np.min(nM))
XLS_sheet.write(XLS_row, 11, stats.scoreatpercentile(nM,25))
XLS_sheet.write(XLS_row, 12, np.median(nM))
XLS_sheet.write(XLS_row, 13, stats.scoreatpercentile(nM,75))
XLS_sheet.write(XLS_row, 14, np.max(nM))

# NUMBER OF OVERESTIMATION OF NUMBER OF PEAKS
XLS_sheet.write(XLS_row, 15, np.mean(nP))
XLS_sheet.write(XLS_row, 16, np.std(nP))
XLS_sheet.write(XLS_row, 17, np.min(nP))
XLS_sheet.write(XLS_row, 18, stats.scoreatpercentile(nP,25))
XLS_sheet.write(XLS_row, 19, np.median(nP))
XLS_sheet.write(XLS_row, 20, stats.scoreatpercentile(nP,75))
XLS_sheet.write(XLS_row, 21, np.max(nP))

# AVERAGE ANGULAR ERROR
XLS_sheet.write(XLS_row, 22, np.mean(AE))
XLS_sheet.write(XLS_row, 23, np.std(AE))
XLS_sheet.write(XLS_row, 24, np.min(AE))
XLS_sheet.write(XLS_row, 25, stats.scoreatpercentile(AE,25))
XLS_sheet.write(XLS_row, 26, np.median(AE))
XLS_sheet.write(XLS_row, 27, stats.scoreatpercentile(AE,75))
XLS_sheet.write(XLS_row, 28, np.max(AE))

# SUM OF FRACTION ERRORS
XLS_sheet.write(XLS_row, 29, np.mean(FE))
XLS_sheet.write(XLS_row, 30, np.std(FE))
XLS_sheet.write(XLS_row, 31, np.min(FE))
XLS_sheet.write(XLS_row, 32, stats.scoreatpercentile(FE, 25))
XLS_sheet.write(XLS_row, 33, np.median(FE))
XLS_sheet.write(XLS_row, 34, stats.scoreatpercentile(FE, 75))
XLS_sheet.write(XLS_row, 35, np.max(FE))

XLS.save("local_measures.xls")