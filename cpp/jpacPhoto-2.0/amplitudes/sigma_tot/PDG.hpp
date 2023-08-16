// Implementation of the HPR1R2 parameterization of hadronic total cross sections
// from the PDG 
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------

#ifndef PDG_HPP
#define PDG_HPP

#include "constants.hpp"
#include "total_xsection.hpp"

namespace jpacPhoto
{
    enum PDG_total_xsections { pipp, pimp };

    class PDG : public raw_total_xsection
    {
        public: 

        PDG(std::array<double, 2> masses, int iso, std::array<double,4> pdgparams)
        : raw_total_xsection(masses)
        {
            _iso   = iso         ;  // isospin sign for pi+ or pi- dependence
            _delta = pdgparams[0];  // VMD/Quark counting paramter for photon 
            _R1    = pdgparams[1];  // Regge terms
            _R2    = pdgparams[2];
            _P     = pdgparams[3];  // Constant pomeron term
        };

        // Only available for on-shell beams so no q2 dependence
        double evaluate(double s, double q2)
        {
            double sab = pow(_mB + _mT + _M, 2);
            return _delta*(_H*pow(log(s/sab), 2) + _P) + _R1*pow(s/sab, -_eta1) - _iso*_R2*pow(s/sab, -_eta2);
        };

        private:

        // Process dependent constants
        int    _iso     = 0;
        double _delta   = 1;
        double _R1 = 0, _R2 = 0, _P = 0;

        // Process independent constants        
        double _M = 2.1206, _H = 0.2720, _eta1 = 0.4473, _eta2 = 0.5486;
    };

    total_xsection new_PDG_sigmatot( PDG_total_xsections x )
    {
        int iso;
        std::array<double,2> masses;
        std::array<double,4> pars;

        switch (x)
        {
            case (pipp): { masses = {M_PION, M_PROTON}; iso = +1; pars = {1, 9.56, 1.767, 18.75}; break; } 
            case (pimp): { masses = {M_PION, M_PROTON}; iso = -1; pars = {1, 9.56, 1.767, 18.75}; break; } 
            default:     { masses = {     0,        0}; iso =  0; pars = {0, 9.56, 1.767, 18.75}; break; } 
        }
        return new_total_xsection<PDG>(masses, iso, pars);
    };
};

#endif