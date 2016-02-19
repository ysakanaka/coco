#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Rank-sum tests table on "Final Data Points".

That is, for example, using 1/#fevals(ftarget) if ftarget was reached
and -f_final otherwise as input for the rank-sum test, where obviously
the larger the better.

One table per function and dimension.

"""
from __future__ import absolute_import

import os, warnings
import numpy
import matplotlib.pyplot as plt
from .. import genericsettings, bestalg, toolsstats, pproc
from ..pptex import tableLaTeX, tableLaTeXStar, writeFEvals2, writeFEvalsMaxPrec, writeLabels
from ..toolsstats import significancetest

from pdb import set_trace

targetsOfInterest = pproc.TargetValues((1e+1, 1e-1, 1e-3, 1e-5, 1e-7))
targetf = 1e-8 # value for determining the success ratio
samplesize = genericsettings.simulated_runlength_bootstrap_sample_size 

table_caption_one = r"""%
    Expected running time (ERT in number of function 
    evaluations) divided by the respective best ERT measured during BBOB-2009 in
    dimensions 5 (left) and 20 (right).
    The ERT and in braces, as dispersion measure, the half difference between 90 and 
    10\%-tile of bootstrapped run lengths appear for each algorithm and 
    """
table_caption_two1 = r"""%
    target, the corresponding best ERT
    in the first row. The different target \Df-values are shown in the top row. 
    \#succ is the number of trials that reached the (final) target $\fopt + 10^{-8}$.
    """
table_caption_two2 = r"""%
    run-length based target, the corresponding best ERT
    (preceded by the target \Df-value in \textit{italics}) in the first row. 
    \#succ is the number of trials that reached the target value of the last column.
    """
table_caption_rest = r"""%
    The median number of conducted function evaluations is additionally given in 
    \textit{italics}, if the target in the last column was never reached. 
    1:\algorithmAshort\ is \algorithmA\ and 2:\algorithmBshort\ is \algorithmB.
    Bold entries are statistically significantly better compared to the other algorithm,
    with $p=0.05$ or $p=10^{-k}$ where $k\in\{2,3,4,\dots\}$ is the number
    following the $\star$ symbol, with Bonferroni correction of #1.
    A $\downarrow$ indicates the same tested against the best algorithm of BBOB-2009.
    """
table_caption = table_caption_one + table_caption_two1 + table_caption_rest
table_caption_expensive = table_caption_one + table_caption_two2 + table_caption_rest

def main(dsList0, dsList1, dimsOfInterest, outputdir, info='', verbose=True):
    """One table per dimension, modified to fit in 1 page per table."""

    #TODO: method is long, split if possible

    dictDim0 = dsList0.dictByDim()
    dictDim1 = dsList1.dictByDim()

    alg0 = set(i[0] for i in dsList0.dictByAlg().keys()).pop().replace(genericsettings.extraction_folder_prefix, '')[0:3]
    alg1 = set(i[0] for i in dsList1.dictByAlg().keys()).pop().replace(genericsettings.extraction_folder_prefix, '')[0:3]

    open(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 'a'
         ).write(r'\providecommand{\algorithmAshort}{%s}' % writeLabels(alg0) + '\n' +
                 r'\providecommand{\algorithmBshort}{%s}' % writeLabels(alg1) + '\n')

    if info:
        info = '_' + info

    bestalgentries = bestalg.loadBestAlgorithm(dsList0.isBiobjective())
    
    header = []
    if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
        header = [r'\#FEs/D']
        headerHtml = ['<thead>\n<tr>\n<th>#FEs/D</th>\n']
        for label in targetsOfInterest.labels():
            header.append(r'\multicolumn{2}{@{}c@{}}{%s}' % label) 
            headerHtml.append('<td>%s</td>\n' % label)
    else:
        header = [r'$\Delta f_\mathrm{opt}$']
        headerHtml = ['<thead>\n<tr>\n<th>&#916; f</th>\n']
        for label in targetsOfInterest.labels():
            header.append(r'\multicolumn{2}{@{\,}c@{\,}}{%s}' % label)
            headerHtml.append('<td>%s</td>\n' % label)
    header.append(r'\multicolumn{2}{@{}l@{}}{\#succ}')
    headerHtml.append('<td>#succ</td>\n</tr>\n</thead>\n')
    
    for d in dimsOfInterest: # TODO set as input arguments
        table = [header]
        tableHtml = headerHtml
        extraeol = [r'\hline']
        try:
            dictFunc0 = dictDim0[d].dictByFunc()
            dictFunc1 = dictDim1[d].dictByFunc()
        except KeyError:
            continue
        funcs = set.union(set(dictFunc0.keys()), set(dictFunc1.keys()))

        nbtests = len(funcs) * 2. #len(dimsOfInterest)

        tableHtml.append('<tbody>\n')
        for f in sorted(funcs):
            tableHtml.append('<tr>\n')
            targets = targetsOfInterest((f, d))
            targetf = targets[-1]
            
            curline = [r'${\bf f_{%d}}$' % f]
            curlineHtml = ['<th><b>f<sub>%d</sub></b></th>\n' % f]

            # generate all data from ranksum test
            entries = []
            ertdata = {}
            for nb, dsList in enumerate((dictFunc0, dictFunc1)):
                try:
                    entry = dsList[f][0] # take the first DataSet, there should be only one?
                except KeyError:
                    warnings.warn('data missing for data set ' + str(nb) + ' and function ' + str(f))
                    print('*** Warning: data missing for data set ' + str(nb) + ' and function ' + str(f) + '***')
                    continue # TODO: problem here!
                ertdata[nb] = entry.detERT(targets)
                entries.append(entry)

            for _t in ertdata.values():
                for _tt in _t:
                    if _tt is None:
                        raise ValueError
                    
            if bestalgentries:            
                bestalgentry = bestalgentries[(d, f)]
                bestalgdata = bestalgentry.detERT(targets)
                bestalgevals, bestalgalgs = bestalgentry.detEvals(targets)
    
                if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
                    # write ftarget:fevals
                    for i in xrange(len(bestalgdata[:-1])):
                        temp = "%.1e" % targetsOfInterest((f, d))[i]
                        if temp[-2]=="0":
                            temp = temp[:-2]+temp[-1]
                        curline.append(r'\multicolumn{2}{@{}c@{}}{\textit{%s}:%s \quad}'
                                       % (temp, writeFEvalsMaxPrec(bestalgdata[i], 2)))
                        curlineHtml.append('<td><i>%s</i>:%s</td>\n' 
                                           % (temp, writeFEvalsMaxPrec(bestalgdata[i], 2)))
                    temp = "%.1e" % targetsOfInterest((f, d))[-1]
                    if temp[-2]=="0":
                        temp = temp[:-2]+temp[-1]
                    curline.append(r'\multicolumn{2}{@{}c@{}|}{\textit{%s}:%s }'
                                   % (temp, writeFEvalsMaxPrec(bestalgdata[-1], 2))) 
                    curlineHtml.append('<td><i>%s</i>:%s</td>\n' 
                                       % (temp, writeFEvalsMaxPrec(bestalgdata[-1], 2))) 
                else:            
                    # write #fevals of the reference alg
                    for i in bestalgdata[:-1]:
                        curline.append(r'\multicolumn{2}{@{}c@{}}{%s \quad}'
                                       % writeFEvalsMaxPrec(i, 2))
                        curlineHtml.append('<td>%s</td>\n' % writeFEvalsMaxPrec(i, 2))
    
                    curline.append(r'\multicolumn{2}{@{}c@{}|}{%s}'
                                   % writeFEvalsMaxPrec(bestalgdata[-1], 2))
                    curlineHtml.append('<td>%s</td>\n' % writeFEvalsMaxPrec(bestalgdata[-1], 2))
    
                tmp = bestalgentry.detEvals([targetf])[0][0]
                tmp2 = numpy.sum(numpy.isnan(tmp) == False)
                curline.append('%d' % (tmp2))
                if tmp2 > 0:
                    curline.append('/%d' % len(tmp))
                    curlineHtml.append('<td>%d/%d</td>\n' % (tmp2, len(tmp)))
                else:
                    curlineHtml.append('<td>%d</td>\n' % (tmp2))
            
            else: # if not bestalgentries
                curline.append(r'\multicolumn{%d}{@{}c@{}|}{}' % (2 * (len(targetsOfInterest.labels()) + 1)))
                curlineHtml.append('<td colspan="%d" />\n' % (len(targetsOfInterest.labels()) + 1))
                
            curlineHtml = [i.replace('$\infty$', '&infin;') for i in curlineHtml]
            table.append(curline[:])
            tableHtml.extend(curlineHtml[:])
            tableHtml.append('</tr>\n')
            extraeol.append('')

            if len(entries) < 2: # funcion not available for *both* algorithms
                continue  # TODO: check which one is missing and make sure that what is there is displayed properly in the following
            
            testres0vs1 = significancetest(entries[0], entries[1], targets)
            
            if bestalgentries:
                testresbestvs1 = significancetest(bestalgentry, entries[1], targets)
                testresbestvs0 = significancetest(bestalgentry, entries[0], targets)

            for nb, entry in enumerate(entries):
                tableHtml.append('<tr>\n')
                if nb == 0:
                    curline = [r'1:\:\algorithmAshort\hspace*{\fill}']
                    curlineHtml = ['<th>1: %s</th>\n' % alg0]
                else:
                    curline = [r'2:\:\algorithmBshort\hspace*{\fill}']
                    curlineHtml = ['<th>2: %s</th>\n' % alg1]

                #data = entry.detERT(targetsOfInterest)
                dispersion = []
                data = []
                evals = entry.detEvals(targets)
                for i in evals:
                    succ = (numpy.isnan(i) == False)
                    tmp = i.copy()
                    tmp[succ==False] = entry.maxevals[numpy.isnan(i)]
                    #set_trace()
                    data.append(toolsstats.sp(tmp, issuccessful=succ)[0])
                    #if not any(succ):
                        #set_trace()
                    if any(succ):
                        tmp2 = toolsstats.drawSP(tmp[succ], tmp[succ==False],
                                                (10, 50, 90), samplesize)[0]
                        dispersion.append((tmp2[-1]-tmp2[0])/2.)
                    else:
                        dispersion.append(None)

                if nb == 0:
                    assert not isinstance(data, numpy.ndarray)
                    data0 = data[:] # TODO: check if it is not an array, it's never used anyway?

                for i, dati in enumerate(data):  

                    z, p = testres0vs1[i] # TODO: there is something with the sign that I don't get
                    # assign significance flag, which is the -log10(p)
                    significance0vs1 = 0
                    if nb != 0:  
                        z = -z  # the test is symmetric
                    if nbtests * p < 0.05 and z > 0:  
                        significance0vs1 = -int(numpy.ceil(numpy.log10(min([1.0, nbtests * p]))))  # this is the larger the more significant

                    isBold = significance0vs1 > 0
                    alignment = 'c'
                    if i == len(data) - 1: # last element
                        alignment = 'c|'

                    if not bestalgentries or numpy.isinf(bestalgdata[i]): # if the 2009 best did not solve the problem

                        tmp = writeFEvalsMaxPrec(float(dati), 2)
                        if not numpy.isinf(dati):
                            if bestalgentries:                        
                                tmpHtml = '<i>%s</i>' % (tmp)
                                tmp = r'\textit{%s}' % (tmp)
                            else:
                                tmpHtml = tmp

                            if isBold:
                                tmp = r'\textbf{%s}' % tmp
                                tmpHtml = '<b>%s</b>' % tmpHtml
                        else:
                            tmpHtml = tmp

                        if dispersion[i] and numpy.isfinite(dispersion[i]):
                            evalsMaxPrec = writeFEvalsMaxPrec(dispersion[i], 1)
                            tmp += r'${\scriptscriptstyle (%s)}$' % evalsMaxPrec
                            tmpHtml += ' (%s)' % evalsMaxPrec
                        tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}' % (alignment, tmp))
                        tableentryHtml = ('%s' % tmpHtml)
                    else:
                        # Formatting
                        tmp = float(dati)/bestalgdata[i]
                        assert not numpy.isnan(tmp)
                        isscientific = False
                        if tmp >= 1000:
                            isscientific = True
                        tableentry = writeFEvals2(tmp, 2, isscientific=isscientific)
                        tableentry = writeFEvalsMaxPrec(tmp, 2)
                        tableentryHtml = writeFEvalsMaxPrec(tmp, 2)

                        if numpy.isinf(tmp) and i == len(data)-1:
                            tableentry = (tableentry 
                                          + r'\textit{%s}' % writeFEvals2(numpy.median(entry.maxevals), 2))
                            tableentryHtml = (tableentryHtml
                                          + ' <i>%s</i>' % writeFEvals2(numpy.median(entry.maxevals), 2))
                            if isBold:
                                tableentry = r'\textbf{%s}' % tableentry
                                tableentryHtml = '<b>%s</b>' % tableentryHtml
                            elif 11 < 3 and significance0vs1 < 0:  # cave: negative significance has no meaning anymore
                                tableentry = r'\textit{%s}' % tableentry
                                tableentryHtml = '<i>%s</i>' % tableentryHtml
                            if dispersion[i] and numpy.isfinite(dispersion[i]/bestalgdata[i]):
                                tableentry += r'${\scriptscriptstyle (%s)}$' % writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)
                                tableentryHtml += ' (%s)' % writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)
                            tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                          % (alignment, tableentry))

                        elif tableentry.find('e') > -1 or (numpy.isinf(tmp) and i != len(data) - 1):
                            if isBold:
                                tableentry = r'\textbf{%s}' % tableentry
                                tableentryHtml = '<b>%s</b>' % tableentryHtml
                            elif 11 < 3 and significance0vs1 < 0:
                                tableentry = r'\textit{%s}' % tableentry
                                tableentryHtml = '<i>%s</i>' % tableentryHtml
                            if dispersion[i] and numpy.isfinite(dispersion[i]/bestalgdata[i]):
                                tableentry += r'${\scriptscriptstyle (%s)}$' % writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)
                                tableentryHtml += ' (%s)' % writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)
                            tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                          % (alignment, tableentry))
                        else:
                            tmp = tableentry.split('.', 1)
                            tmpHtml = tableentryHtml.split('.', 1)
                            if isBold:
                                tmp = list(r'\textbf{%s}' % i for i in tmp)
                                tmpHtml = list('<b>%s</b>' % i for i in tmpHtml)
                            elif 11 < 3 and significance0vs1 < 0:
                                tmp = list(r'\textit{%s}' % i for i in tmp)
                                tmpHtml = list('<i>%s</i>' % i for i in tmpHtml)
                            tableentry = ' & .'.join(tmp)
                            tableentryHtml = '.'.join(tmpHtml)
                            if len(tmp) == 1:
                                tableentry += '&'
                            if dispersion[i] and numpy.isfinite(dispersion[i]/bestalgdata[i]):
                                tableentry += r'${\scriptscriptstyle (%s)}$' % writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)
                                tableentryHtml += ' (%s)' % writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)

                    superscript = ''
                    superscriptHtml = ''

                    if bestalgentries:
                        if nb == 0:
                            z, p = testresbestvs0[i]
                        else:
                            z, p = testresbestvs1[i]
    
                        #The conditions are now that ERT < ERT_best
                        if ((nbtests * p) < 0.05 and dati - bestalgdata[i] < 0.
                            and z < 0.):
                            nbstars = -numpy.ceil(numpy.log10(nbtests * p))
                            #tmp = '\hspace{-.5ex}'.join(nbstars * [r'\star'])
                            if z > 0:
                                superscript = r'\uparrow' #* nbstars
                                superscriptHtml = '&uarr;'
                            else:
                                superscript = r'\downarrow' #* nbstars
                                superscriptHtml = '&darr;'
                                # print z, linebest[i], line1
                            if nbstars > 1:
                                superscript += str(int(nbstars))
                                superscriptHtml += str(int(nbstars))

                    if superscript or significance0vs1:
                        s = ''
                        shtml = ''
                        if significance0vs1 > 0:
                            s = '\star'
                            shtml = '&#9733;'
                        if significance0vs1 > 1:
                            s += str(significance0vs1)
                            shtml += str(significance0vs1)
                        s = r'$^{' + s + superscript + r'}$'
                        shtml = '<sup>' + shtml + superscriptHtml + '</sup>' 

                        if tableentry.endswith('}'):
                            tableentry = tableentry[:-1] + s + r'}'
                        else:
                            tableentry += s
                        tableentryHtml += shtml

                    tableentryHtml = tableentryHtml.replace('$\infty$', '&infin;')                
                    curlineHtml.append('<td>%s</td>\n' % tableentryHtml)
                    curline.append(tableentry)

                    #curline.append(tableentry)
                    #if dispersion[i] is None or numpy.isinf(bestalgdata[i]):
                        #curline.append('')
                    #else:
                        #tmp = writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 2)
                        #curline.append('(%s)' % tmp)

                tmp = entry.evals[entry.evals[:, 0] <= targetf, 1:]
                try:
                    tmp = tmp[0]
                    curline.append('%d' % numpy.sum(numpy.isnan(tmp) == False))
                    curlineHtml.append('<td>%d' % numpy.sum(numpy.isnan(tmp) == False))
                except IndexError:
                    curline.append('%d' % 0)
                    curlineHtml.append('<td>%d' % 0)
                curline.append('/%d' % entry.nbRuns())
                curlineHtml.append('/%d</td>\n' % entry.nbRuns())

                table.append(curline[:])
                tableHtml.extend(curlineHtml[:])
                tableHtml.append('</tr>\n')
                extraeol.append('')

            extraeol[-1] = r'\hline'
        extraeol[-1] = ''

        outputfile = os.path.join(outputdir, 'pptable2_%02dD%s.tex' % (d, info))
        spec = r'@{}c@{}|' + '*{%d}{@{}r@{}@{}l@{}}' % len(targetsOfInterest) + '|@{}r@{}@{}l@{}'
        res = r'\providecommand{\algorithmAshort}{%s}' % writeLabels(alg0) + '\n'
        res += r'\providecommand{\algorithmBshort}{%s}' % writeLabels(alg1) + '\n'
        # open(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 'a').write(res)
        
        #res += tableLaTeXStar(table, width=r'0.45\textwidth', spec=spec,
                              #extraeol=extraeol)
        res += tableLaTeX(table, spec=spec, extraeol=extraeol)
        f = open(outputfile, 'w')
        f.write(res)
        f.close()
        
        res = ("").join(str(item) for item in tableHtml)
        res = '<p><b>%d-D</b></p>\n<table>\n%s</table>\n' % (d, res)

        filename = os.path.join(outputdir, genericsettings.two_algorithm_file_name + '.html')
        lines = []
        with open(filename) as infile:
            for line in infile:
                if '<!--pptable2Html-->' in line:
                    lines.append(res)
                lines.append(line)
                
        with open(filename, 'w') as outfile:
            for line in lines:
                outfile.write(line)     

        if verbose:
            print "Table written in %s" % outputfile

