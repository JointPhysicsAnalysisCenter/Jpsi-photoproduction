// Implementations of angular functions
// This includes Wigner-d functions with respect to both angle theta and cosTheta
// and the ususal Legendre functions
//
// Author:       Daniel Winney (2020)
// Affiliation:  Joint Physics Analysis Center (JPAC)
// Email:        dwinney@iu.edu
// ---------------------------------------------------------------------------

#ifndef ANGULAR_FUNCTIONS_HPP
#define ANGULAR_FUNCTIONS_HPP

#include <iostream>
#include <complex>
#include <algorithm>

#include "constants.hpp"

namespace jpacPhoto
{
    // ---------------------------------------------------------------------------
    // Wigner d-func coefficient of leading power
    double wigner_leading_coeff(int j, int lam1, int lam2);

    // Wigner d-function for half-integer spin
    double wigner_d_half(int j, int lam1, int lam2, double theta);

    // Wigner d-function for integer spin
    double wigner_d_int(int j, int lam1, int lam2, double theta);

    // Wigner d-function for integer spin in terms of the cosine of theta not theta
    complex wigner_d_int_cos(int j, int lam1, int lam2, double cos);

    // Legendre function in terms of cosine theta
    double legendre(int l, double z);
};

#endif
