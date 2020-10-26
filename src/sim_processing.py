# -*- coding: utf-8 -*-
"""
Post processing routines.

@author: A. Zech
Licence MIT, A.Zech, 2020
"""

import numpy as np
from scipy.integrate import cumtrapz

def long_average(points, values, coord="x"):

    """ Average given values defined at points (x,y,z) along a chosen coordinate.
    Default setting gives a longitudinal average where values are averaged
    over y and z coordinate.

    Definition
    ----------
    arg_list, long_average    =    long_average(points,values)

    Input
    ----------
    points      :   array
                    list of coordinates (same length as array of values)
    valuees     :   array
                    values of quantity at coordinates (same length as 'points')

    Output
    ---------
    arg_list    :   array
                    locations x of averaged values
    ens_data    :   array
                    averaged quantity C(x)
       """

    if len(points) != len(values):
        print("dimension of points and values do not match")

    if coord == "x":
        co = 0
    elif coord == "y":
        co = 1
    elif coord == "z":
        co = 2
    else:
        raise ValueError("Coordinate not defined! Choose between 'x', 'y' or 'z'.")

    arg_list = np.unique(points[:, co])
    long_average = np.zeros(len(arg_list))

    for ii, xx in enumerate(arg_list):
        ID = np.where(points[:, co] == xx)
        long_average[ii] = np.average(values[ID])

    return arg_list, long_average


class RealizationMass:

    """
    Class for postprocessing simulated mass distribution by OGS
        - calculate norm values
            self.norm_values()
        - exclude negative values:
            self.non_negative()
        - recalcuate norm & normalize
            self.norm_values()
            self.normalize()
        - check for convergence:
            self.convergence()
        - aggregate values to specified length scales:
            self.aggregate(agg)
        - calculate cumulative mass distribution (cdf)
            self.cumulate()
        - interpolate cdf along standard values for infering quantiles
            self.cdf_interpolate()
        - write results to csv file
            self.write_results(file_name)
    """

    def __init__(self, mass, times, x, normalize=True):

        self.mass = np.array(mass, ndmin=2, dtype=float)
        self.x = np.array(x, ndmin=1, dtype=float)
        self.t = np.array(times, ndmin=1, dtype=float)

        if self.mass.shape == (len(self.x), len(self.t)):
            self.mass.transpose()
        elif self.mass.shape != (len(self.t), len(self.x)):
            raise ValueError("dimensions of mass, time and x-coordinate do not match")

        if normalize:
            self.non_negative()
            self.norm_values()
            self.normalize()
        else:
            self.norm_values()
        self.agg = False
        self.mass_cum = None

    def norm_values(self):

        """ Calculate norm value of mass distribution """

        self.norm = np.zeros_like(self.t)
        for i in range(len(self.t)):
            self.norm[i] = np.trapz(self.mass[i, :], x=self.x)

        return self.norm

    def non_negative(self):

        """ remove negative values (as result of numerical oscillation)  """

        self.mass = np.where(self.mass >= 0, self.mass, 0)

    def normalize(self):

        """ Normalize mass """

        self.mass = self.mass / np.tile(self.norm, (len(self.x), 1)).T

    def aggregate(self, agg, loc0=0, normalize=True):

        """ Aggregate data to arguments at aggregation level  """

        if agg < 0:
            raise ValueError("Invalid aggregation level lower then 0")
        elif agg < min(np.diff(self.x)):
            raise ValueError("Invalid aggregation level smaller then dx")
        elif int(agg) == int(np.mean(np.diff(self.x))):
            print("Data already at aggregation level")
        else:
            self.agg = agg

        loc = min(loc0, self.x[0])
        cxt_agg = []
        x_agg = []

        while loc <= self.x[-1]:

            index = np.where(  # index of C-values to average for specific x-location
                abs(self.x - loc) <= 0.5 * self.agg
            )[0]

            if len(index) == 0:
                cx_agg = np.zeros(self.mass.shape[0])
            else:
                cx_agg = np.mean(self.mass[:, index], axis=1)

            cxt_agg.append(cx_agg)
            x_agg.append(loc)
            loc = loc + self.agg
        cxt_agg.append(np.zeros_like(cx_agg))
        x_agg.append(loc)

        self.mass = np.array(cxt_agg).T
        self.x = np.array(x_agg)

        if normalize:
            self.norm = self.agg * np.sum(self.mass, axis=1)
            self.normalize()

        return self.x, self.mass

    def convergence(self, sill1=-1, sill2=1000):

        """ check for convergence """

        check_mass = self.mass / np.tile(self.norm, (len(self.x), 1)).T

        test01 = np.any(np.isnan(self.norm))  ### nan-value in norm
        test02 = np.any(np.isnan(check_mass))  ### nan-values in mass distribution

        test11 = np.any(self.norm < 0)  ### negative norm
        test12 = np.any(check_mass < sill1)  ### high negative mass values

        test21 = np.any(self.norm > sill2)  ### exceptionally high norm
        test22 = np.any(check_mass > sill2)  ### exceptionally high mass values

        self.converged = test01 and test02 and test11 and test12 and test21 and test22

        return self.converged

    def cumulate(self):

        """ Calculates CDF of data (cumulating data values)    """

        if self.agg > 0:
            self.mass_cum = self.agg * np.cumsum(self.mass, axis=1)
            self.x_cum = self.x + 0.5 * self.agg
        else:  ### integration for non-aggregated data
            dx = np.zeros_like(self.x)
            dx[:-1] = np.diff(self.x)
            dx[-1] = np.mean(np.diff(self.x))
            self.x_cum = self.x + 0.5 * dx
            self.mass_cum = cumtrapz(self.mass, self.x, axis=1, initial=0)

        return self.x_cum, self.mass_cum

    def cdf_interpolate(self, cr=np.arange(0, 1, 0.01)):

        """ interpolation of cumulated data to standard range [0,1] """

        if self.mass_cum is None:
            self.cumulate()

        ### recovered mass of interest
        xr = np.zeros((len(self.t), len(cr)))
        ### interpolate location from data

        for it in range(len(self.t)):
            xr[it, :] = np.interp(cr, self.mass_cum[it, :], self.x_cum)

        return xr, cr

    def write_results(self, file_name=False):

        """
            summarizing mass and coordinates into one matrix
            for saving simulation results of a realization to a file
        """
        write_results = np.zeros((len(self.t) + 1, len(self.x) + 1))
        write_results[1:, 0] = self.t
        write_results[0, 1:] = self.x
        write_results[1:, 1:] = self.mass
        if file_name:
            np.savetxt(file_name, write_results, delimiter=",")

        return write_results
