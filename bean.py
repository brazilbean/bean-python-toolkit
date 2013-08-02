## Bean.py - Gordon Bean's bag-o-tricks
# brazilbean@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy import stats as pystat

import time
import collections
import threading
import types
import os
import re

# ---------------------- #
#   Miscelaneous Stuff   #
# ---------------------- #

# Parameters 
# Eventually I may move this to a class...
def default_param( params, **defaults ):
    if params is None or defaults is None:
        return dict()
    for key, value in defaults.iteritems():
        if key not in params:
            params[key] = value
    return params

class Timer:
    start = None
    def __init__(self):
        self.start = None
    def tic(self):
        self.start = time.time()
    def toc(self, start=None):
        if start is None:
            t = time.time() - self.start
        else:
            t = time.time() - start
        print "%f seconds elapsed\n"%t
        return t

_t__ = Timer()
tic = _t__.tic
toc = _t__.toc

def dirfiles( dirname, pattern='.*', includepath=True, flags=0 ):
    """ List the contents of a directory matching the regex pattern. """
    if dirname[-1] is not os.sep:
        dirname += os.sep
    files = [f for f in os.listdir(dirname) if re.search(pattern, f, flags)]
    files.sort()
        
    if includepath:
        if dirname[0] is os.sep:
            path = dirname
        else:
            path = os.getcwd() + os.sep + dirname
        files = [path + f for f in files]
    
    return files
    
def verbose( verb, msg ):
    if verb:
        print msg
        
def filescan( filename, pattern, types, headerlines = 0 ):
    with open(filename, 'r') as fid:
        # Skip header lines
        for h in range(0, headerlines):
            fid.readline()
        
        # Read data
        tmp = [ [t(x) for t,x in zip(types,re.search(pattern, line).groups()) ] 
            for line in fid ]
        
    return zip(*tmp)
    
## Multi-threading
class MapThread( threading.Thread ):
    def __init__(self, fun, data, out, domain):
        threading.Thread.__init__(self)
        self.fun = fun
        self.data = data
        self.out = out
        self.domain = domain
        
    def run(self):
        if self.out is None:
            map(self.fun, self.data[self.domain])
        else:
            self.out[self.domain] = map(self.fun, self.data[self.domain])

def multimap( fun, data, output=None, numthreads=2 ):
    """ A multi-threaded version of map. """
    # Determine number of elements per thread
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        
    n = len(data)
    ii = np.ceil(np.linspace(0, 10, numthreads+1)).astype(int)
    
    # Set up threads
    threads = [ MapThread( fun, data, output, range(ii[i-1],ii[i])) \
        for i in range(1,numthreads+1)]
        
    # Run threads
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
#----------------------------#
#   Index and matrix stuff   #
# ---------------------------#
def nar( foo ):
    """ Make a 2D Numpy Array """
    if isinstance(foo, np.ndarray):
        if len(foo.shape) == 1:
            # Make it a 1 x N array
            tmp = np.ndarray((1,len(foo)))
            tmp[:] = foo
            return tmp
        else:
            return np.array(foo)
    else:
        # Make it a 1 x N array
        tmp = np.ndarray((1,len(foo)))
        tmp[:] = foo
        return tmp
        
def tr( foo ):
    if len(foo.shape) == 1:
        return np.transpose(nar(foo))
    else:
        return np.transpose(foo)
        
# MATLAB find
def find( array, which=None ):
    # Array is numpy array, logical values
    inds = np.nonzero(array);
    if (which is None):
        return inds[0]
    elif (len(inds[0]) > which):
        return inds[0][which]
    else:
        return None;

# IN from mfiles
def ind( data, index=None, nx1=False, copy=False ):
    if copy:
        data = np.array(data)
        
    if index is None:
        # Vectorize data
        if nx1:
            return np.reshape( data, (np.prod(data.shape),1) )
        else:
            return np.reshape( data, np.prod(data.shape) )
    else:
        # Return data[index]
        if hasattr(index, '__call__'):
            # Index is a function
            return data[index(data)]
        else:
            # Assume index is numeric
            return data[index]

def nisnan(data):
    return ~np.isnan(data)
    
# FIL from mfiles
def fil( data, filt, fill = np.nan ):
    # Assumes data is a numpy.ndarray
    data = np.array(data); # Make a copy

    # If the filter is a function
    if hasattr(np.isnan, '__call__'): #isinstance(filt, types.FunctionType):
        data[filt(data)] = fill
    elif type(filt) is np.ndarray and filt.dtype is np.dtype(bool):
        if filt.shape == data.shape:
            data[filt] = fill
        else:
            raise Exception('Boolean filter must have the same shape as data')
            
    return data
        
# NaNs
def nans( shape, dtype=float ):
    tmp = np.empty(shape, dtype)
    tmp.fill( np.nan ) 
    return tmp

# Nan ignoring hist
def nanhist(data, *args, **kwargs):
    data = data[~np.isnan(data)]
    return plt.hist(data, *args, **kwargs)
        
# Numel
def numel( data ):
    return np.prod(data.shape)
    
def ind2sub( ii, shape ):
    if np.any(ii >= np.prod(shape)):
        raise Exception('Index exceeds provided dimensions: %i in (%i,%i)'% \
            (ii[find(ii>=np.prod(shape),0)], shape[0], shape[1]))
    return ii / shape[1], np.mod(ii, shape[1])
    
def _firstdim(data):
    # Data is numpy array
    dim = find(np.array(data.shape) != 1, 0);
    if dim is None:
        return 1;
    else:
        return dim;

def ipermute(data, dims):
    # Data is numpy array, dims is list of dimensions
    maxdim = np.max(dims)+1;
    dims = np.array(dims);
    idims = tuple(find(dims==ii,0) for ii in range(0,maxdim));
    return np.transpose(data, idims);

def _reshapedim(data, dim):
    ndims = len(data.shape);
    todim = np.hstack( (dim, np.setdiff1d(range(0,ndims),(dim,))));
    
    data_ = np.transpose(data, tuple(x for x in todim))
    sizes = data_.shape;
    data_ = np.reshape( data_, (sizes[0], np.prod(sizes[1:])));
    return data_;

def _ireshapedim(data_, shape, dim):
    ndims = len(shape);
    todim = np.hstack( (dim, np.setdiff1d(range(0,ndims),(dim,))));

    shape = [shape[ii] for ii in todim];
    shape[0] = 1;
    shape = tuple(x for x in shape);
    
    data_ = np.reshape( data_, shape );
    return ipermute( data_, todim ); 
    
#---------------------#
#   Statistic stuff   #
#---------------------#
def nanmean(data, dim=None):
    if dim is None:
        data = ind(data, nx1=True)
        dim = 0
    
    # Remove nans
    data = np.copy(data);
    nans = np.isnan(data);
    data[nans] = 0;
    
    # Compute mean
    snans = np.array(np.sum(~nans,dim)+0.0);
    snans[snans==0] = np.nan;
    return np.sum(data,dim) / snans;
    
def nanvar(data, flag=0, dim=None):
    if dim is None:
        dim = _firstdim(data)
    
    data_ = _reshapedim( data, dim )
    dm = nanmean(data_,0)
    dnan = np.isnan(data_)
    
    derr = (data_ - dm)**2
    derr[dnan] = 0
    
    tmp = (np.sum(~dnan,0)-1+flag);
    tmp[tmp==0] = 1;
    
    nv = np.sum(derr,0) / tmp
    nv[ np.sum(~dnan,0)==1 ] = 0
    nv[ np.sum(~dnan,0)==0 ] = np.nan
    
    return _ireshapedim( nv, data.shape, dim )
    
def nanstd(data, flag=0, dim=None):
    return np.sqrt( nanvar( data, flag, dim ) );
    
def pmode( data, width=None ):
    # Data should be a numpy array, treated as a vector
    if len(data.shape) > 1:
        data = ind(data)
    
    if width is None:
        width = 1.06 * nanstd(data) * np.sum(~np.isnan(data))**(-0.2)

    n = np.sum(~np.isnan(data));
    if n == 0:
        return np.nan
    elif n == 1:
        return data[~np.isnan(data)]
    else:
        x0 = np.min(data)
        xF = np.max(data)
        if np.mod(xF-x0,2) == 1:
            xF = xF + 1
            
        x = np.linspace(x0, xF, n);
        
        # Discritize data
        y = np.histogram( data, x );
        y = y[0];
        
        # Convolve with standard normal
        g = pystat.norm(0,width).pdf(x - np.median(x));
        score = np.convolve( y, g );
        
        # Find max
        list2 = find(score == np.max(score)) - np.floor(len(g)/2)
        list2 = [ x[max(min(ii,len(x)),1)] for ii in list2 ]
        
        return np.mean(list2)
        #list2 = np.argmax( score )
        
        # Account for convolution shift
        #list2 = list2 - np.floor(len(g)/2);
        #if isinstance( list2, collections.Iterable ):
        #    list2 = [ max(min(ii,len(x)),1) for ii in list2 ]
        #else:
        #    list2 = max(min(ii,len(x)),1)
        #    
        ##list2[ list2 > len(x) ] = len(x);
        ##list2[ list2 < 1 ] = 1;
        ##list2 = [ii for ii in list2]
        #
        ## Return peak
        #return np.mean( x[list2] );
    
def pmodend( data, dim=None ):
    # Data should be a numpy array

    # Find dimension
    if dim is None:
        dim = _firstdim(data);
    
    data_ = _reshapedim( data, dim );
    
    center = np.zeros(data_.shape[1])
    for ii in range(0, data_.shape[1]):
        center[ii] = pmode( data_[:,ii] );

    return _ireshapedim( center, data.shape, dim );

#--------------------------#
#   Image analysis stuff   #
#--------------------------#
def centroids( imgb ):
    # imgb is logical numpy array
    # Get labels
    imgl = ndimage.label( imgb );
    stats = np.zeros((imgl[1],2))
    for ii in range(0,imgl[1]):
        stats[ii,:] = ndimage.center_of_mass( imgb, imgl[0], ii);
    return stats;
    
def areas( imgb ):
    # imgb is logical numpy array
    # Get labels
    imgl = ndimage.label( imgb );
    stats = np.zeros(imgl[1])
    for ii in range(0,imgl[1]):
        stats[ii] = ndimage.sum( imgb, imgl[0], ii);
    return stats;

def imagesc( data, interpolation='none', **kwargs ):
    plt.imshow( data, interpolation=interpolation, **kwargs )
    
def qline( x, y, *args, **kwargs):
    if isinstance(x, collections.Iterable):
        x = [z for z in x]
    else:
        x = [x, x]
    
    if isinstance(y, collections.Iterable):
        y = [z for z in y]
    else:
        y = [y, y]
    
    xl = plt.xlim()
    yl = plt.ylim()
    plt.plot(x, y, *args, **kwargs)
    plt.xlim(xl)
    plt.ylim(yl)

def get_points( npoints, **plotargs ):
    default_param( plotargs, 
        markersize=20, 
        marker='+',
        color='k' )
    presspoints = []
    points = []
    def onpress(event):
        presspoints.append((event.xdata, event.ydata))
    def onrelease(event):
        point = (event.xdata, event.ydata)
        if point == presspoints[-1]:
            xl, yl = plt.xlim(), plt.ylim()
            points.append(point)
            plt.plot(point[0], point[1], **plotargs)
            plt.xlim(xl), plt.ylim(yl)
            plt.draw()
            
    cidrelease = plt.gcf().canvas.mpl_connect('button_release_event', onrelease)
    cidpress = plt.gcf().canvas.mpl_connect('button_press_event', onpress)
    
    while len(points) < npoints:
        plt.pause(0.01)
    plt.gcf().canvas.mpl_disconnect(cidrelease)
    plt.gcf().canvas.mpl_disconnect(cidpress)
    
    return np.array(points)

