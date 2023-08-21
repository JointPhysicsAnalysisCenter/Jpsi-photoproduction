// Wigner little-d functions for half-integer spin particles
//
// Author:       Daniel Winney (2020)
// Affiliation:  Joint Physics Analysis Center (JPAC)
// Email:        dwinney@iu.edu
// ---------------------------------------------------------------------------

#include "angular_functions.hpp"

namespace jpacPhoto
{

    double legendre(int l, double z)
    {
        switch (l) 
        {
            case 0: return 1.;
            case 1: return z;
            case 2: return 0.5*(3.*z*z - 1.);
            case 3: return 0.5*z*(5.*z*z - 3.);
            case 4: return (35.*z*z*z*z - 30.*z*z + 3.)/8.;
            case 5: return z*(63.*z*z*z*z - 70.*z*z + 15.)/8.;
            default:
            {
                error("legendre", "L value " + std::to_string(l) + " not available! Returning 0.", 0.);
            }
        };

        return 0.;
    };

    // --------------------------------------------------------------------------
    double wigner_leading_coeff(int j, int lam1, int lam2)
    {
        int M = std::max(std::abs(lam1), std::abs(lam2));
        int N = std::min(std::abs(lam1), std::abs(lam2));

        int lambda = std::abs(lam1 - lam2) + lam1 - lam2;

        double result = (double) factorial(2*j);
        result /= sqrt( (double) factorial(j-M));
        result /= sqrt( (double) factorial(j+M));
        result /= sqrt( (double) factorial(j-N));
        result /= sqrt( (double) factorial(j+N));
        result /= pow(2.,  double(j-M));
        result *= pow(-1., double(lambda)/2.);

        return result;
    };

    // ---------------------------------------------------------------------------
    // USING WIKIPEDIA SIGN CONVENTION
    // theta is in radians
    // lam1 = 2 * lambda and lam2 = 2 * lambda^prime are integers
    double wigner_d_half(int j, int lam1, int lam2, double theta)
    {
        double phase = 1.;
        if ( j % 2 == 0 || (lam1 + lam2) % 2 != 0 )
        {
            error("wigner_d_half", "Invalid arguments passed! Returning 0.", 0);
        };

        // If first lam argument is smaller, switch them
        if (abs(lam1) < abs(lam2))
        {
            int temp = lam1;
            lam1 = lam2;
            lam2 = temp;

            phase *= pow(-1., double((lam1 - lam2)/ 2));
        };

        // If first lam is negative, switch them
        if (lam1 < 0)
        {
            lam1 *= -1;
            lam2 *= -1;

            phase *= pow(-1., double((lam1 - lam2)/ 2));
        }

        double result = 0.;

        int id = ((lam2 > 0) - (lam2 < 0)) * (j * 100 + lam1 * 10 + abs(lam2)); // negative sign refers to negative lam2
        switch (id)
        {
            // spin 1/2 
            case  111: 
            {
                result =  cos(theta / 2.); 
                break;
            };
            case -111: 
            {
                result = -sin(theta / 2.); 
                break;
            };

            // spin 3/2
            case  333:       
            {
                result = cos(theta / 2.) / 2.;
                result *= (1. + cos(theta));
                break;
            }
            case  331:
            {
                result = - sqrt(3.) / 2.;
                result *= sin(theta / 2.);
                result *= 1. + cos(theta);
                break;
            }
            case -331:
            {
                result = sqrt(3.) / 2.;
                result *= cos(theta / 2.);
                result *= 1. - cos(theta);
                break;
            }
            case -333:
            {
                result = - sin(theta / 2.) / 2.;
                result *= 1. - cos(theta);
                break;
            }
            case  311:
            {
                result = 1. / 2.;
                result *= 3. * cos(theta) - 1.;
                result *= cos(theta / 2.);
                break;
            }
            case -311:
            {
                result = -1. / 2.;
                result *= 3. * cos(theta) + 1.;
                result *= sin(theta / 2.);
                break;
            }

            // Spin- 5/2
            case  533:
            {
                result = -1. / 4.;
                result *= cos(theta / 2.);
                result *= (1. + cos(theta)) * (3. - 5. * cos(theta));
                break;
            }
            case  531:
            {
                result = sqrt(2.) / 4.;
                result *= sin(theta / 2.);
                result *= (1. + cos(theta)) * (1. - 5. * cos(theta));
                break;
            }
            case -531:
            {
                result =  sqrt(2.) / 4.;
                result *= cos(theta / 2.);
                result *= (1. - cos(theta)) * (1. + 5. * cos(theta));
                break;
            }
            case -533:
            {
                result = -1. / 4.;
                result *= sin(theta / 2.);
                result *= (1. - cos(theta)) * (3. + 5. * cos(theta));
                break;
            }
            case  511:
            {
                result = -1. / 2.;
                result *= cos(theta / 2.);
                result *= (1. + 2. * cos(theta) - 5. * cos(theta)*cos(theta));
                break;
            }
            case -511:
            {
                result = 1. / 2.;
                result *= sin(theta / 2.);
                result *= (1. - 2. * cos(theta) - 5. * cos(theta)*cos(theta));
                break;
            }

            default: return 0.;
        };

        return phase * result;
    };

    // ---------------------------------------------------------------------------
    double wigner_d_int(int j, int lam1, int lam2, double theta)
    {

        double phase = 1.;
        // If first lam argument is smaller, switch them
        if (abs(lam1) < abs(lam2))
        {
            int temp = lam1;
            lam1 = lam2;
            lam2 = temp;

            phase *= pow(-1., double(lam1 - lam2));
        };

        // If first lam is negative, smitch them
        if (lam1 < 0)
        {
            lam1 *= -1;
            lam2 *= -1;

            phase *= pow(-1., double(lam1 - lam2));
        }

        // Output
        double result = 0.;

        int id = ((lam2 >= 0) - (lam2 < 0)) * (j * 100 + lam1 * 10 + abs(lam2)); // negative sign refers to negative lam2
        switch (id)
        {   
            // Spin 1
            case  111:
            {
                result = (1. + cos(theta)) / 2.;
                break;
            }
            case  110:
            {
                result = - sin(theta) / sqrt(2.);
                break;
            }
            case -111:
            {
                result = (1. - cos(theta)) / 2.;
                break;
            }
            case  100:
            {
                result = cos(theta);
                break;
            }

            default: return 0.;
        }

        return phase * result;
    };

    // ---------------------------------------------------------------------------
    complex wigner_d_int_cos(int j, int lam1, int lam2, double cosine)
    {
        // Careful because this loses the +- phase of the sintheta. 
        complex sine = sqrt(XR - cosine * cosine);

        complex sinhalf =  sqrt((XR - cosine) / 2.);
        complex coshalf =  sqrt((XR + cosine) / 2.);

        double phase = 1.;
        // If first lam argument is smaller, switch them
        if (abs(lam1) < abs(lam2))
        {
            int temp = lam1;
            lam1 = lam2;
            lam2 = temp;

            phase *= pow(-1., double(lam1 - lam2));
        };

        // If first lam is negative, smitch them
        if (lam1 < 0)
        {
            lam1 *= -1;
            lam2 *= -1;

            phase *= pow(-1., double(lam1 - lam2));
        }

        complex result = 0.;
        int id = ((lam2 >= 0) - (lam2 < 0)) * (j * 100 + lam1 * 10 + abs(lam2)); // negative sign refers to negative lam2
        switch (id)
        {   
            // Spin 1
            case  111:
            {
                result = (1. + cosine) / 2.;
                break;
            }
            case  110:
            {
                result = - sine / sqrt(2.);
                break;
            }
            case -111:
            {
                result = (1. - cosine) / 2.;
                break;
            }
            case  100:
            {
                result = cosine;
                break;
            }

            default: return 0.;
        }

        return phase * result;
    };
};