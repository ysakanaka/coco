#! /usr/bin/env python

"""Creates run length distribution figures."""


from __future__ import absolute_import

import os
import numpy
import matplotlib.pyplot as plt
from pdb import set_trace
from bbob_pproc import bootstrap

#__all__ = []

rldColors = ('k', 'c', 'm', 'r', 'k', 'c', 'm', 'r', 'k', 'c', 'm', 'r')
rldUnsuccColors = ('k', 'c', 'm', 'k', 'c', 'm', 'k', 'c', 'm', 'k', 'c', 'm')  # should not be too short
# Used as a global to store the largest xmax and align the FV ECD figures.
fmax = None
evalfmax = None

def plotECDF(x, n=None, plotArgs={}):
    if n is None:
        n = len(x)
    nx = len(x)
    if n == 0 or nx == 0:
        res = plt.plot([], [], **plotArgs)
    else:
        x2 = numpy.hstack(numpy.repeat(sorted(x), 2))
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, nx) / float(n), 2),
                           float(nx)/n])
        res = plt.plot(x2, y2, **plotArgs)
    return res

def beautifyECDF(axish=None):
    if axish is None:
        axish = plt.gca()
    plt.ylim(0.0, 1.0)
    axish.grid('True')

def beautifyRLD(figHandle, figureName, maxEvalsF, fileFormat=('png', 'eps'),
                text=None, verbose=True):
    """Format the figure of the run length distribution and save into files."""
    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')
    plt.axvline(x=maxEvalsF, color='k')
    plt.xlim(1.0, maxEvalsF ** 1.05)
    axisHandle.set_xlabel('log10 of FEvals / DIM')
    axisHandle.set_ylabel('proportion of trials')
    # Grid options
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(numpy.log10(j)))
    axisHandle.set_xticklabels(newxtic)

    beautifyECDF()

    plt.text(0.5, 0.93, text, horizontalalignment="center",
             transform=axisHandle.transAxes)
             #bbox=dict(ec='k', fill=False), 

    #set_trace()
    plt.legend(loc='best')
    #if legend:
        #axisHandle.legend(legend, locLegend)

    # Save figure
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 300,
                    format = entry)
        if verbose:
            print 'Wrote figure in %s.' %(figureName + '.' + entry)



def plotRLDistr(indexEntries, fvalueToReach, maxEvalsF, verbose=True):
    """Creates run length distributions from a sequence of indexEntries.

    Keyword arguments:
    indexEntries
    fvalueToReach
    verbose

    Outputs:
    res -- resulting plot.
    fsolved -- number of different functions solved.
    funcs -- number of different function considered.
    """

    x = []
    nn = 0
    fsolved = set()
    funcs = set()
    for i in indexEntries:
        funcs.add(i.funcId)
        for j in i.hData:
            if j[0] <= fvalueToReach[i.funcId]:
                #This loop is needed because though some number of function
                #evaluations might be below maxEvals, the target function value
                #might not be reached yet. This is because the horizontal data
                #do not go to maxEvals.

                for k in range(1, i.nbRuns() + 1):
                    if j[i.nbRuns() + k] <= fvalueToReach[i.funcId]:
                        x.append(j[k] / i.dim)
                        fsolved.add(i.funcId)
                break
        nn += i.nbRuns()

    # For the label the last i.funcId is used.
    kwargs = plotArgs.copy()
    kwargs['label'] = (kwargs.get('label', '') +
                       ('%+d:%d/%d' % (numpy.log10(fvalueToReach[i.funcId]),
                                      len(fsolved), len(funcs))))
    
    #res = plotECDF(x, nn, kwargs)
    n = len(x)
    if n == 0:
        res = plt.plot([], [], **kwargs)
    else:
        x.sort()
        x2 = numpy.hstack([numpy.repeat(x, 2), maxEvalsF ** 1.05])
        # maxEvalsF: used for the limit of the plot
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, n+1)/float(nn), 2)])
        res = plt.plot(x2, y2, **kwargs)

    return res#, fsolved, funcs


def plotRLDistr2(dsList, fvalueToReach, maxEvalsF, plotArgs={},
                 verbose=True):
    """Creates run length distributions from a sequence dataSetList.

    Keyword arguments:
    dataSetList
    fvalueToReach
    verbose

    Outputs:
    res -- resulting plot.
    fsolved -- number of different functions solved.
    funcs -- number of different function considered.
    """

    x = []
    nn = 0
    fsolved = set()
    funcs = set()
    for i in dsList:
        funcs.add(i.funcId)
        for j in i.evals:
            if j[0] <= fvalueToReach[i.funcId]:
                #set_trace()
                tmp = j[1:]
                x.extend(tmp[numpy.isfinite(tmp)]/float(i.dim))
                fsolved.add(i.funcId)
                #TODO: what if j[numpy.isfinite(j)] is empty
                break
        nn += i.nbRuns()

    #set_trace()
    #if len(x) > 0:
        #x.append(maxEvalsF ** 1.05)
    # For the label the last i.funcId is used.
    kwargs = plotArgs.copy()
    #set_trace()
    try:
        kwargs['label'] = kwargs.setdefault('label',
                          ('%+d:%d/%d' % (numpy.log10(fvalueToReach[i.funcId]),
                                          len(fsolved), len(funcs))))
    except TypeError: # fvalueToReach == 0. for instance...
        pass

    #res = plotECDF(x, nn, kwargs)
    n = len(x)
    if n == 0:
        res = plt.plot([], [], **kwargs)
    else:
        x.sort()
        x2 = numpy.hstack([numpy.repeat(x, 2), maxEvalsF ** 1.05])
        # maxEvalsF: used for the limit of the plot
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, n+1)/float(nn), 2)])
        res = plt.plot(x2, y2, **kwargs)

    return res#, fsolved, funcs

def plotERTDistr(dsList, fvalueToReach, plotArgs=None, verbose=True):
    """Creates estimated run time distributions from a sequence dataSetList.

    Keyword arguments:
    dataSetList
    fvalueToReach
    verbose

    Outputs:
    res -- resulting plot.
    fsolved -- number of different functions solved.
    funcs -- number of different function considered.
    """

    x = []
    nn = 0
    samplesize = 1000 # samplesize is at least 1000
    percentiles = 0.5 # could be anything...

    for i in dsList:
        #funcs.add(i.funcId)
        for j in i.evals:
            if j[0] <= fvalueToReach[i.funcId]:
                runlengthsucc = j[1:][numpy.isfinite(j[1:])]
                runlengthunsucc = i.maxevals[numpy.isnan(j[1:])]
                tmp = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                       percentiles=percentiles,
                                       samplesize=samplesize)
                x.extend(tmp[1])
                break
        nn += samplesize
    #set_trace()
    res = plotECDF(x, nn, plotArgs)

    return res

def beautifyFVD(figHandle, figureName, fileFormat=('png','eps'),
                isStoringXMax=False, text=None, verbose=True):
    """Formats the figure of the run length distribution.

    Keyword arguments:
    isStoringMaxF -- if set to True, the first call BeautifyVD sets the global
                     fmax and all subsequent call will have the same maximum
                     xlim.
    """

    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')

    if isStoringXMax:
        global fmax
    else:
        fmax = None

    if not fmax:
        xmin, fmax = plt.xlim()
    plt.xlim(1., fmax)

    #axisHandle.invert_xaxis()
    axisHandle.set_xlabel('log10 of Df / Dftarget')
    # axisHandle.set_ylabel('proportion of successful trials')
    # Grid options
    beautifyECDF()

    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(numpy.log10(j)))
    axisHandle.set_xticklabels(newxtic)
    axisHandle.set_yticklabels(())

    plt.text(0.98, 0.02, text, horizontalalignment="right",
             transform=axisHandle.transAxes)
             #bbox=dict(ec='k', fill=False), 

    # Save figure
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 300,
                    format = entry)
        if verbose:
            print 'Wrote figure in %s.' %(figureName + '.' + entry)

def plotFVDistr(indexEntries, fvalueToReach, maxEvalsF, verbose=True):
    """Creates empirical cumulative distribution functions of final function
    values plot from a sequence of indexEntries.

    Keyword arguments:
    indexEntries -- sequence of IndexEntry to process.
    fvalueToReach -- float used for the lower limit of the plot
    maxEvalsF -- indicates which vertical data to display.
    verbose -- controls verbosity.

    Outputs: a plot of a run length distribution.
    """

    x = []
    nn = 0
    for i in indexEntries:
        for j in i.vData:
            if j[0] >= maxEvalsF * i.dim:
                break
        x.extend(j[i.nbRuns()+1:] / fvalueToReach[i.funcId])
        nn += i.nbRuns()

    res = plotECDF(x, nn)

    return res

def plotFVDistr2(dataSetList, fvalueToReach, maxEvalsF, plotArgs={},
                 verbose=True):
    """Creates empirical cumulative distribution functions of final function
    values plot from a sequence of indexEntries.

    Keyword arguments:
    indexEntries -- sequence of IndexEntry to process.
    fvalueToReach -- float used for the lower limit of the plot
    maxEvalsF -- indicates which vertical data to display.
    verbose -- controls verbosity.

    Outputs: a plot of a run length distribution.
    """

    x = []
    nn = 0
    for i in dataSetList:
        for j in i.funvals:
            if j[0] >= maxEvalsF * i.dim:
                break
        #set_trace()
        tmp = j[1:].copy() / fvalueToReach[i.funcId]
        tmp[tmp==0] = 1. # HACK
        x.extend(tmp)
        nn += i.nbRuns()

    #set_trace()
    res = plotECDF(x, nn, plotArgs)

    return res

def main(indexEntries, valuesOfInterest, isStoringXMax=False, outputdir='',
         info='default', verbose=True):
    """Generate figures of empirical cumulative distribution functions.

    Keyword arguments:
    indexEntries -- list of IndexEntry instances to process.
    valuesOfInterest -- target function values to be displayed.
    isStoringXMax -- if set to True, the first call BeautifyVD sets the globals
                     fmax and maxEvals and all subsequent calls will use these
                     values as rightmost xlim in the generated figures.
     -- if set to True, the first call BeautifyVD sets the global
                     fmax and all subsequent call will have the same maximum
                     xlim.
    outputdir -- output directory (must exist)
    info --- string suffix for output file names.

    Outputs:
    Image files of the empirical cumulative distribution functions.
    """

    #sortedIndexEntries = sortIndexEntries(indexEntries)

    plt.rc("axes", labelsize=20, titlesize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("font", size=20)
    plt.rc("legend", fontsize=20)

    maxEvalsFactor = max(i.mMaxEvals()/i.dim for i in indexEntries)
    #maxEvalsFactorCeil = numpy.power(10,
                                     #numpy.ceil(numpy.log10(maxEvalsFactor)))

    if isStoringXMax:
        global evalfmax
    else:
        evalfmax = None

    if not evalfmax:
        evalfmax = maxEvalsFactor

    figureName = os.path.join(outputdir,'pprldistr%s' %('_' + info))
    fig = plt.figure()
    legend = []
    for j in range(len(valuesOfInterest)):
        tmp = plotRLDistr(indexEntries, valuesOfInterest[j], evalfmax,
                          verbose)
        #set_trace()
        if not tmp is None:
            plt.setp(tmp, 'color', rldColors[j])
            #set_trace()
            #legend.append('%+d:%d/%d' %  
                          #(numpy.log10(valuesOfInterest[j]), len(fsolved), 
                           #len(f)))
            if rldColors[j] == 'r':  # 1e-8 in bold
                plt.setp(tmp, 'linewidth', 3)

    funcs = list(i.funcId for i in indexEntries)
    if len(funcs) > 1:
        text = 'f%d-%d' %(min(funcs), max(funcs))
    else:
        text = 'f%d' %(funcs[0])

    beautifyRLD(fig, figureName, evalfmax, text=text, verbose=verbose)
    plt.close(fig)

    figureName = os.path.join(outputdir,'ppfvdistr_%s' %(info))
    fig = plt.figure()
    for j in range(len(valuesOfInterest)):
        #set_trace()
        tmp = plotFVDistr(indexEntries, valuesOfInterest[j],
                          evalfmax, verbose=verbose)
        #if not tmp is None:
        plt.setp(tmp, 'color', rldColors[j])
        if rldColors [j] == 'r':  # 1e-8 in bold
            plt.setp(tmp, 'linewidth', 3)

    tmp = numpy.floor(numpy.log10(evalfmax))
    # coloring left to right:
    #maxEvalsF = numpy.power(10, numpy.arange(tmp, 0, -1) - 1)
    # coloring right to left:
    maxEvalsF = numpy.power(10, numpy.arange(0, tmp))

    #The last index of valuesOfInterest is still used in this loop.
    #set_trace()
    for k in range(len(maxEvalsF)):
        tmp = plotFVDistr(indexEntries, valuesOfInterest[j],
                          maxEvalsF=maxEvalsF[k], verbose=verbose)
        plt.setp(tmp, 'color', rldUnsuccColors[k])

    beautifyFVD(fig, figureName, text=text, isStoringXMax=isStoringXMax,
                verbose=verbose)

    plt.close(fig)

    plt.rcdefaults()
