// Implementation of total cross-section from JPAC's  2015 paper [1]
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------
// REFERENCES:
// [1] - https://arxiv.org/abs/1506.01764
// ------------------------------------------------------------------------------

#ifndef JPAC_PIN_HPP
#define JPAC_PIN_HPP

#include "cgamma.hpp"
#include "constants.hpp"
#include "data_set.hpp"
#include "total_xsection.hpp"
#include <Math/Interpolator.h>

namespace jpacPhoto
{
    class JPAC_piN : public raw_total_xsection
    {
        public: 

        // Iso here refers to +-1 the sign of the PRODUCES MESON
        // this is the opposite of the exchanged pion
        JPAC_piN(int iso)
        : raw_total_xsection({M_PION, M_PROTON}), _iso(iso)
        {
            for (int L = 0; L <= _Lmax; L++)
            {
                _pws.push_back(std::make_shared<SAID_PW>(L));
            };
        };

        JPAC_piN(int iso, int L)
        : raw_total_xsection({M_PION, M_PROTON}), _iso(iso)
        {
            _pws.push_back(std::make_shared<SAID_PW>(L));
        };

        // Only available for on-shell beams so no q2 dependence
        inline double evaluate(double s, double q2)
        {
            double cutoff = 4.2;
            if ( sqrt(s) < (M_PROTON + M_PION) ) return 0;

            double x1 = cutoff;
            double x2 = cutoff + 1;

            if ( s <= x1 )
            {
                return resonances(s, q2);
            }
            else if ( (s > x1) && (s < x2) )
            {
                double fx1 = resonances(x1, q2);
                double fx2 = regge(x2);
                double y  = (s - x1) / (x2 - x1);
                return fx1 *(1 - y) + fx2 * y;
            }
            else return regge(s);
        };

        private:

        int _iso = 0;
            
        // Lab beam momentum
        inline static double pLab(double s)
        {
            double Elab = (s - M2_PION - M2_PROTON) / (2.*M_PROTON); 
            return sqrt(Elab*Elab - M2_PION);
        };

        // ----------------------------------------------------------------------
        // Low-energy pieces

        class SAID_PW
        {
            public: 
            SAID_PW(int L)
            : _L(L)
            {
                import();
            };

            // Since we only care about the sigma_tot we save only imag of each PW
            inline double imag(int IJ, double s)
            {
                // There is no 2L-1 wave for L = 0
                if (_L == 0 && IJ < 0) return 0;

                double plab = pLab(s);
                switch (IJ)
                {
                    case (+1): return _imag_1p.Eval(plab);
                    case (-1): return _imag_1m.Eval(plab);
                    case (+3): return _imag_3p.Eval(plab);
                    case (-3): return _imag_3m.Eval(plab);
                    default: return 0;
                }
            };

            // Calculate the L'th contribution to sigma_tot
            inline double sigma(int iso, double s)
            {
                double W    = sqrt(s);
                double E    = (s + M2_PROTON - M2_PION) / (2.*W);
                double Elab = (s - M2_PROTON - M2_PION) / (2.*M_PROTON);
                double qcm  = sqrt(Kallen(s, M2_PROTON, M2_PION)) / (2.*W);

                double f1 = 0., f3 = 0., g1 = 0., g3 = 0.;
                double fp, fm, gp, gm;
                double f1p, f1m, f2p, f2m;
                double Ap, Am, Bp, Bm, Cp, Cm;


                // Assemble amplitudes from PWAs
                f1 = ((_L+1)*imag(+1, s) + (_L)*imag(-1, s));
                f3 = ((_L+1)*imag(+3, s) + (_L)*imag(-3, s));
                g1 = _L*(_L+1)/2 * (imag(+1, s) - imag(-1, s));
                g3 = _L*(_L+1)/2 * (imag(+3, s) - imag(-3, s));

                f1 /= qcm; f3 /= qcm; g1 /= qcm; g3 /= qcm;

                // Use these to calculate the t-channel amplitudes
                fp = (f1 + 2.*f3) / 3.;
                gp = (g1 + 2.*g3) / 3.;
                fm = (f1 - f3)    / 3.;
                gm = (g1 - g3)    / 3.;


                // Amplitudes f1 and f2 with t-channel isospin
                f1p = fp + gp;
                f2p = -gp;
                f1m = fm + gm;
                f2m = -gm;

                // Invariant amplitudes
                Ap = (W+M_PROTON)/(E+M_PROTON)*f1p - (W-M_PROTON)/(E-M_PROTON)*f2p;
                Am = (W+M_PROTON)/(E+M_PROTON)*f1m - (W-M_PROTON)/(E-M_PROTON)*f2m;

                Bp = f1p/(E+M_PROTON) + f2p/(E-M_PROTON);
                Bm = f1m/(E+M_PROTON) + f2m/(E-M_PROTON);

                Cp = 4*M_PI * (Ap + Elab * Bp);
                Cm = 4*M_PI * (Am + Elab * Bm);

                return 0.389352 * (Cp - double (iso) * Cm) / pLab(s);
            };

            inline int L(){ return _L; };

            private:

            // Quantum numbers
            int _L; 

            inline void import()
            {
                std::string filepath = "/data/said/SAID_PiN_";
                
                auto pw_1p = import_data<9>(filepath + std::to_string(_L) + std::to_string(1) + std::to_string(2*_L+1) + ".txt");
                auto pw_3p = import_data<9>(filepath + std::to_string(_L) + std::to_string(3) + std::to_string(2*_L+1) + ".txt");
                
                _imag_1p.SetData( pw_1p[0]*1E-3, pw_1p[6]);
                _imag_3p.SetData( pw_3p[0]*1E-3, pw_3p[6]);

                if (_L != 0)
                {
                    auto pw_1m = import_data<9>(filepath + std::to_string(_L) + std::to_string(1) + std::to_string(2*_L-1) + ".txt");
                    auto pw_3m = import_data<9>(filepath + std::to_string(_L) + std::to_string(3) + std::to_string(2*_L-1) + ".txt");

                    _imag_1m.SetData( pw_1m[0]*1E-3, pw_1m[6]);
                    _imag_3m.SetData( pw_3m[0]*1E-3, pw_3m[6]);
                }
            };

            // Saved interpolations of each imaginary part
            ROOT::Math::Interpolator _imag_1m, _imag_1p, _imag_3m, _imag_3p;
        };

        int _Lmax = 7, _Lcut = 3;
        std::vector<std::shared_ptr<SAID_PW>> _pws;

        inline double resonances(double s, double q2)
        {
            double bfpi  = Kallen(s, q2, M2_PROTON) / Kallen(s, M2_PION, M2_PROTON);
            double sum = 0;
            for (auto wave : _pws)
            {
                int L = (wave->L() < _Lcut) ? wave->L() : _Lcut;
                if (L >= 3 && s < 1.18) continue;
                if (L >= 4 && s < 1.3)  continue;

                double pw = wave->sigma(_iso, s);
                if (pw < 0) continue;

                sum += pow(bfpi, L) * pw ;
            };

            return sum;
        };

        // ----------------------------------------------------------------------
        // High-energy pieces

        // Internal class object for a Regge pole
        class regge_pole
        {
            public:

            regge_pole(int sig, std::array<double,2> pars)
            : _tau(sig), 
            _alpha0(pars[0]), _c(pars[1])
            {};

            inline double operator()(double s)
            {
                int n     = (_tau + 1)/2;
                double nu = (s - M2_PROTON - M2_PION) / (2*M_PROTON); 
                complex signature_factor = (1+_tau*exp(-I*PI*_alpha0))/2;
                return - std::imag(_c * signature_factor * pow(nu / sqrt(_s0), _alpha0) * cgamma(n-_alpha0));

            };

            private:

            int    _tau = +1;
            double _c = 0, _alpha0 = 0;
            double _s0 = 1;
        };

        // Three poles contribute
        regge_pole _rho = regge_pole(-1, {0.49,   5.01});
        regge_pole _f   = regge_pole(+1, {0.49,  71.35});
        regge_pole _pom = regge_pole(+1, {1.075, 23.89});

        inline double regge(double s)
        {
            double isoscalar = _pom(s) + _f(s), isovector = _rho(s);
            return 0.389352 * (isoscalar - _iso * isovector) / pLab(s);
        };
    };
};

#endif