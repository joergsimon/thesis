import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.linear_model as lm
import sklearn.neighbors.nearest_centroid as nc
import sklearn.neighbors as ne
import sklearn.naive_bayes as nb
import sklearn.ensemble as em
import sklearn.discriminant_analysis as da
import sklearn.gaussian_process as gp
from sklearn.base import clone
from sklearn import svm
from sklearn import tree
from sklearn.metrics import f1_score
import numpy as np
import pickle
import sklearn.metrics as metr
import glob
import pandas as pd
import datetime
import os

def get_svc_triple(type, decision_fun, kernel, gamma, degree, c_or_nu):
    if gamma == 'auto':
        name = f"{type}[{kernel}-{decision_fun},c/nu{c_or_nu:.2},gauto,p{degree}]"
    else:
        name = f"{type}[{kernel}-{decision_fun},c/nu{c_or_nu:.2},g{gamma:.2},p{degree}]"
    if type == "SVC":
        s = svm.SVC(C=c_or_nu, gamma=gamma, kernel=kernel, degree=degree, decision_function_shape=decision_fun, class_weight='balanced')
    else:
        s = svm.NuSVC(nu=c_or_nu, gamma=gamma, kernel=kernel, degree=degree, decision_function_shape=decision_fun, class_weight='balanced')
    return (name, s, True)

class SVMGenerator:
    
    def __init__(self):
        self.kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        self.decicion_fun = ['ovo', 'ovr']
        self.gammarange = (1e-2,1e2)
        self.polyrange = (2,4)
        self.crange = (1e-2,1e2)
        self.nurange = (1e-100,1)
    
    def get_random_svm(self, g=None):
        dfun = decicion_fun[np.random.randint(0,len(self.decicion_fun))]
        kernel = kernels[np.random.randint(0,len(self.kernels))]
        if nu_or_c > 0.5: # C
            return self.get_random_svc(dfun, kernel, g=g)
        else: # Nu
            return self.get_random_nusvc(dfun, kernel, g=g)
    
    def get_random_nusvc(self, dfun, kernel, g=None):
        gr, pr, cr, nr = self.gammarange, self.polyrange, self.crange, self.nurange
        nu_or_c = np.random.random_sample()
        gamma = np.random.uniform(gr[0],gr[1])
        if g == "auto":
            gamma = g
        nc = np.random.uniform(nr[0],nr[1])
        s = get_svc_triple("NuSVC" , dfun, kernel, gamma, poly, nc)
        return s
    
    def get_random_svc(self, dfun, kernel, g=None):
        gr, pr, cr, nr = self.gammarange, self.polyrange, self.crange, self.nurange
        nu_or_c = np.random.random_sample()
        gamma = np.random.uniform(gr[0],gr[1])
        if g == "auto":
            gamma = g
        nc = np.random.uniform(cr[0],cr[1])
        s = get_svc_triple("SVC" , dfun, kernel, gamma, poly, nc)
        return s
        

def get_n_random_svms(n, generator=SVMGenerator()):
    result = []
    for i in range(n):
        s = generator.get_random_svm()
        result.append(s)
    return result

def get_random_svm_grid(range=3, generator=SVMGenerator()):
    def make_tuples(arr):
        res = []
        i = 1
        while i < len(arr):
            res.append((arr[i-1], arr[i]))
        return res
    result = []
    cr = make_tuples(np.linspace(generator.crange[0], generator.crange[1], num=range+1))
    nr = make_tuples(np.linspace(generator.nurange[0], generator.nurange[1], num=range+1))
    dr = make_tuples(np.linspace(generator.polyrange[0], generator.polyrange[1], num=range+1))
    gr = make_tuples(np.linspace(generator.gammarange[0], generator.gammarange[1], num=range+1))
    for df in generator.decicion_fun:
        for kernel in generator.kernels:
            if kernel in ['rbf', 'poly', 'sigmoid']:
                if kernel == 'poly':
                    for degreer in dr:
                        for gammar in gr:
                            for c in cr:
                                generator.gammarange = gammar
                                generator.polyrange = degreer
                                generator.crange = c
                                s = generator.get_random_svc(df, kernel)
                                result.append(s)
                            for nu in nr:
                                generator.gammarange = gammar
                                generator.polyrange = degreer
                                generator.nurange = nu
                                s = generator.get_random_nusvc(df, kernel)
                                result.append(s)
                        for c in cr:
                            generator.polyrange = degreer
                            generator.crange = c
                            s = generator.get_random_svc(df, kernel, g="auto")
                            result.append(s)
                        for nu in nr:
                            generator.polyrange = degreer
                            generator.nurange = nu
                            s = generator.get_random_nusvc(df, kernel, g="auto")
                            result.append(s)
                else:
                    for gamma in gr:
                        for c in cr:
                            generator.gammarange = gammar
                            generator.crange = c
                            s = generator.get_random_svc(df, kernel)
                            result.append(s)
                        for nu in nr:
                            generator.gammarange = gammar
                            generator.nurange = nu
                            s = generator.get_random_nusvc(df, kernel)
                            result.append(s)
                    for c in cr:
                        generator.crange = c
                        s = generator.get_random_svc(df, kernel, g="auto")
                        result.append(s)
                    for nu in nr:
                        generator.nurange = nu
                        s = generator.get_random_nusvc(df, kernel, g="auto")
                        result.append(s)
                    
            else:
                for c in cr:
                    generator.crange = c
                    s = generator.get_random_svc(df, kernel, g="auto")
                    result.append(s)
                for nu in nr:
                    generator.nurange = nu
                    s = generator.get_random_nusvc(df, kernel, g="auto")
                    result.append(s)
    return result

def get_svm_grid(range=3, generator=SVMGenerator()):
    result = []
    # first SVC then NuSVC:
    cr = np.linspace(generator.crange[0], generator.crange[1], num=range)
    nr = np.linspace(generator.nurange[0], generator.nurange[1], num=range)
    dr = np.linspace(generator.polyrange[0], generator.polyrange[1], num=range)
    gr = np.linspace(generator.gammarange[0], generator.gammarange[1], num=range)
    for df in generator.decicion_fun:
        for kernel in generator.kernels:
            if kernel in ['rbf', 'poly', 'sigmoid']:
                if kernel == 'poly':
                    for degree in dr:
                        for gamma in gr:
                            for c in cr:
                                s = get_svc_triple("SVC", df, kernel, gamma, degree, c)
                                result.append(s)
                            for nu in nr:
                                if nu >= 0.5 and degree == 4.0:
                                    continue
                                s = get_svc_triple("NuSVC", df, kernel, gamma, degree, nu)
                                result.append(s)
                        for c in cr:
                            s = get_svc_triple("SVC", df, kernel, 'auto', degree, c)
                            result.append(s)
                        for nu in nr:
                            s = get_svc_triple("NuSVC", df, kernel, 'auto', degree, nu)
                            result.append(s)
                else:
                    for gamma in gr:
                        for c in cr:
                            s = get_svc_triple("SVC", df, kernel, gamma, 0, c)
                            result.append(s)
                        for nu in nr:
                            s = get_svc_triple("NuSVC", df, kernel, gamma, 0, nu)
                            result.append(s)
                    for c in cr:
                        s = get_svc_triple("SVC", df, kernel, 'auto', degree, c)
                        result.append(s)
                    for nu in nr:
                        s = get_svc_triple("NuSVC", df, kernel, 'auto', degree, nu)
                        result.append(s)
                    
            else:
                for c in cr:
                    s = get_svc_triple("SVC", df, kernel, 'auto', 0, c)
                    result.append(s)
                for nu in nr:
                    s = get_svc_triple("NuSVC", df, kernel, 'auto', 0, nu)
                    result.append(s)
    return result
    