import numpy as np 
import dask
import timeit

def predict_over_time(x, y, z, overlay=False):
    "Predicts a quantity at times = 0, 1, ... 14"
    out = np.zeros((x.shape[0], 15))
    for t in range(15):
        out[:, t] = t * x ** 2 + y - 2 * z - 2 * t
    adj = 1.5 if overlay else 1.0
    return adj * out

from numba import jit


@jit
def jitted_func(x, y, z, overlay=False):
    "Predicts a quantity at times = 0, 1, ... 14"
    out = np.zeros((x.shape[0], 15))
    for t in range(15):
        out[:, t] = t * x ** 2 + y - 2 * z - 2 * t
    adj = 1.5 if overlay else 1.0
    return adj * out

  
# create some artificial inputs
n = 25000
u = np.random.random(n)
x = np.random.poisson(lam=5, size=n)
y, z = np.random.normal(size=(n, 2)).T


timeit for n in range(100):
	_ = predict_over_time(x, y, z) # 100 loops, best of 3: 3.28 ms per loop


#%%timeit -n 100
_ = jitted_func(x, y, z) # 100 loops, best of 3: 2.27 ms per loop

from numba import guvectorize


@guvectorize('i8, f8, f8, b1, f8[:], f8[:]',
             '(), (), (), (), (s) -> (s)')
def fast_predict_over_time(x, y, z, overlay, _, out):
    adj = 1.5 if overlay else 1.0
    for t in range(len(out)):
        out[t] = adj * (t * x ** 2 + y - 2 * z - 2 * t)

res = np.zeros((n, 15))

#%%timeit -n 100
_ = fast_predict_over_time(x, y, z, False, res) # 100 loops, best of 3: 575 µs per loop

from dask import delayed


# won't be evaluated until we call .compute()
fast_predict_over_time = delayed(fast_predict_over_time)

## using the same numpy arrays from above...

#%%timeit -n 100
_ = fast_predict_over_time(x, y, z, False, res).compute()
# 100 loops, best of 3: 1.04 ms per loop

@delayed
def predict_another_thing(x, y, z):
    # model scoring code goes here
    pass


@delayed
def complicated_feature_b(x, y):
    # lets imagine this feature is *very* expensive to create
    # and takes a full minute to process
    sleep(60); return x * y


@delayed
def feature_a(u, x):
    return 20 * u - x


## put our delayed objects into a dictionary for easy access   
results = {'feature_a': feature_a(u, x),
           'feature_b': complicated_feature_b(x, z)}
results['predict_another_thing'] = predict_another_thing(y, x, results['feature_a'])
results['no_overlay'] = fast_predict_over_time(x, results['feature_a'], results['feature_b'], False, res)
results['w_overlay'] = fast_predict_over_time(x, z, results['feature_a'], True, res)

#%%timeit -n 100
_, _ = dask.compute(results['w_overlay'], 
                    results['predict_another_thing'])
# 100 loops, best of 3: 1.61 ms per loop
