// Implementation of a PWA in the scattering-length approximation with up to three
// coupled channels
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef K_MATRIX_HPP
#define K_MATRIX_HPP

#include "constants.hpp"
#include "kinematics.hpp"
#include "partial_wave.hpp"

namespace jpacPhoto
{
    namespace analytic
    {
        class K_matrix : public raw_partial_wave
        {
            public: 

            // Single channel K-matrix
            K_matrix(amplitude_key key, kinematics xkinem, int J, std::string id = "K_matrix")
            : raw_partial_wave(key, xkinem, J, id)
            {
                _Nth = 1;
                initialize(2);

                // Populate the thresholds
                _thresholds[0] = {xkinem->get_meson_mass(), xkinem->get_recoil_mass()};
            };

            // Two-channel K-matrix
            K_matrix(amplitude_key key, kinematics xkinem, int J, std::array<double,2> masses, std::string id = "K_matrix")
            : raw_partial_wave(key, xkinem, J, id)
            {
                _Nth = 2;
                initialize(5);

                // Populate the thresholds
                _thresholds[0] = {xkinem->get_meson_mass(), xkinem->get_recoil_mass()};
                _thresholds[1] = masses;
            };

            // Three-channel K-matrix
            K_matrix(amplitude_key key, kinematics xkinem, int J, std::array<std::array<double,2>,2> masses, std::string id = "K_matrix")
            : raw_partial_wave(key, xkinem, J, id)
            {
                _Nth = 3;
                initialize(9);

                // Populate the thresholds
                _thresholds[0] = {xkinem->get_meson_mass(), xkinem->get_recoil_mass()};
                _thresholds[1] = masses[0];
                _thresholds[2] = masses[1];
            };

            // -----------------------------------------------------------------------
            // Virtuals 

            // These are projections onto the orbital angular momentum and therefore
            // helicity independent
            inline complex helicity_amplitude(std::array<int,4> helicities, double s, double t)
            {
                // Save inputes
                store(helicities, s, t);
                
                return (_debug == 1) ? (2*_J+1) * legendre(_J, _kinematics->z_s(s,t)) * partial_wave(_s)
                                     : (2*_J+1) * legendre(_J, cos(_theta))           * partial_wave(_s);
            };

            // We can have any quantum numbers
            inline std::vector<particle> allowed_mesons() { return any(); };
            inline std::vector<particle> allowed_baryons(){ return any(); };

            // And helicity independent
            inline helicity_frame native_helicity_frame(){ return HELICITY_INDEPENDENT; };

            // Parameter names are a[J] and b[J] for scattering length and normalization respectively
            inline std::vector<std::string> parameter_labels()
            {
                std::vector<std::string> labels;
                switch (_Nth)
                {
                    case 1: { labels = { J_label("n"), J_label("a") }; break; }
                    case 2: { labels = labels_2C; break; }
                    case 3: { labels = labels_3C; break; }
                };

                // If EffectiveRange, add extra labels
                switch (_option)
                {
                    case kScatteringLength: return labels;
                    case kEffectiveRange:  
                    {
                        if (_Nth == 1)
                        {
                            labels.push_back( J_label("b"));
                            return labels;
                        };

                        labels.push_back( J_label("b00") );
                        labels.push_back( J_label("b11") );
                        if (_Nth == 2) return labels;

                        labels.push_back( J_label("b22") );
                        return labels;
                    }
                    default:  return {{}};
                };
            };

            // The amplitude_option to choose a limiting case of parameters
            // or Default to reset to the most general parameterization
            inline void set_option(int x)
            {
                switch (x)
                {
                    case kScatteringLength:
                    {
                        set_N_pars(2+(_Nth>1)*(4*_Nth-5));
                        _b00 = 0; _b11 = 0; _b22 = 0; break;
                    };
                    case kEffectiveRange:   
                    { 
                        set_N_pars(3+(_Nth>1)*(5*_Nth-6)); break;
                    };
                    default: option_error(); return;
                };

                _option = x;
            };

            // PW is the unitarized K-matrix form
            inline complex partial_wave(double s)
            {
                store({_lamB, _lamT, _lamX, _lamR}, s, _t);

                // Recalculate production related quantities for each threshold
                for (int i = 0; i < _Nth; i++)
                {
                    _q[i] = q(i);  // Break up momenta
                    _G[i] = G(i);  // Phase-space factor
                }

                // K-matrices elements
                _K00 = pow(q2(0,0), _J) * (_a00 + _b00*q2(0,0));

                _K01 = pow(q2(0,1), _J) *  _a01;
                _K11 = pow(q2(1,1), _J) * (_a11 + _b11*q2(1,1));

                _K02 = pow(q2(0,2), _J) *  _a02;
                _K12 = pow(q2(1,2), _J) *  _a12;
                _K22 = pow(q2(2,2), _J) * (_a22 + _b22*q2(2,2));

                // Determinant of K-matrix
                _detK = _K00*_K11*_K22 + 2*_K01*_K02*_K12 - _K02*_K02*_K11 - _K12*_K12*_K00 - _K01*_K01*_K22;

                // The M-matrices all share the same denominator    
                _D = (1+_G[0]*_K00)*(1+_G[1]*_K11)*(1+_G[2]*_K22) - _G[0]*_G[1]*_K01*_K01
                                                                  - _G[0]*_G[2]*_K02*_K02
                                                                  - _G[1]*_G[2]*_K12*_K12 - _G[0]*_G[1]*_G[2]*(_K00*_K11*_K22 - _detK);

                // Then all we need are the numerators
                _N[0] =   pow(pq(0), _J) * _n[0] * ((1 + _G[1]*_K11)*(1 + _G[2]*_K22) - _G[1]*_G[2]*_K12*_K12);
                _N[1] = - pow(pq(1), _J) * _n[1] * _G[1]*(_K01 + _G[2]*(_K01*_K22 - _K02*_K12));
                _N[2] = - pow(pq(2), _J) * _n[2] * _G[2]*(_K02 + _G[1]*(_K02*_K11 - _K01*_K12));

                return (_N[0] + _N[1] + _N[2]) / _D;
            };

            // Extract just the elastic amplitude as a function of s
            inline complex elastic_amplitude(double s)
            {
                partial_wave(s);

                // Determinants of the sub-matrices
                complex detK01, detK02, detK12;
                detK01 = _K00*_K11 - _K01*_K01;
                detK02 = _K00*_K22 - _K02*_K02;
                detK12 = _K11*_K22 - _K12*_K12;

                // Use this to calculate the elastic 00 element of T-matrix 
                complex N    = _K00 + _G[1]*detK01 + _G[2]*detK02 + _G[1]*_G[2]*(detK12*_K00 + detK02*_K11 + detK01*_K22 - 2*_K00*_K11*_K22 + 2*_K01*_K02*_K12);
                return N / _D;
            };

            // Ratio of direct vs indirect photoproduction
            inline double zeta()
            {
                partial_wave(_kinematics->sth());

                double direct   = std::abs(_N[0]);
                double indirect = std::abs(_N[1] + _N[2]);

                return indirect / (direct + indirect);
            };

            inline double scattering_length()
            {
                double T00th = std::real( elastic_amplitude(_kinematics->sth() ));
                double SL    = - T00th / (8*PI*_kinematics->Wth());
                return SL / 5.068; // Output in units of fm
            };

            inline double R_VMD()
            {
                double F = std::real( partial_wave(     _kinematics->sth()) );
                double T = std::real( elastic_amplitude(_kinematics->sth()) );
                double gVMD = 0.0273;

                return std::abs(F / T) / gVMD;
            };

            inline double total_xsection(double s)
            {
                if (s <= _kinematics->sth()) return 0;

                // Output in mb!
                return  0.389352 * (2*_J+1)*std::imag(elastic_amplitude(s)) / sqrt(Kallen(_s, _mX*_mX, _mR*_mR));
            };

            // Options
            static const int kScatteringLength = 0;
            static const int kEffectiveRange   = 1;

            protected:

            inline void allocate_parameters(std::vector<double> pars)
            {
                switch (_Nth)
                {
                    default: return;
                    case 1: 
                    {
                        _n[0] = pars[0];
                        _a00  = pars[1];
                        break;
                    };
                    case 2:
                    {
                        _n[0] = pars[0];
                        _n[1] = pars[1];
                        _a00  = pars[2];
                        _a01  = pars[3];
                        _a11  = pars[4];
                        break;
                    };
                    case 3:
                    {
                        _n[0] = pars[0];
                        _n[1] = pars[1];
                        _n[2] = pars[2];
                        _a00  = pars[3];
                        _a01  = pars[4];
                        _a02  = pars[5];
                        _a11  = pars[6];
                        _a12  = pars[7];
                        _a22  = pars[8];
                        break;
                    };
                };

                if ( _option == kEffectiveRange )
                {
                    _b00 = pars[2+(_Nth>1)*(4*_Nth-5)];
                    _b11 = (_Nth > 1) ? pars[4*_Nth-2] : 0;
                    _b22 = (_Nth > 2) ? pars[4*_Nth-1] : 0;
                }
            };

            // If we use three channels
            int _Nth = 0;

            // Mass of intermediate coupled channels
            // we can have up to two additional channels
            std::array<std::array<double,2>,3> _thresholds;
            
            // Production parameters
            std::array<complex,3> _q = {0,0,0}, _G = {0,0,0}, _N = {0,0,0};

            // K-matrix parameters
            double _a00 = 0, _a01 = 0, _a02 = 0;
            double           _a11 = 0, _a12 = 0;
            double                     _a22 = 0;

            // Second order parameters (effective range)
            double _b00 = 0, _b11 = 0, _b22 = 0;
            
            // Production amplitude parameters
            std::array<double,3> _n = {0,0,0};

            // Internal variables for the K matrices
            complex _K00 = 0, _K01 = 0, _K02 = 0;
            complex           _K11 = 0, _K12 = 0;
            complex                     _K22 = 0;
            complex  _D = 0,  _detK = 0;

            // Chew-Mandelstam phase-space 
            inline complex G(double m1, double m2)
            {
                complex rho, xi;
                complex result;

                rho    = csqrt(Kallen(_s, m1*m1, m2*m2)) / _s;
                xi     = 1 - (m1+m2)*(m1+m2)/_s;
                result = (rho*log((xi + rho) / (xi - rho)) - xi*(m2-m1)/(m2+m1)*log(m2/m1)) / PI;
                return result / (16*PI);
            };
            inline complex G(unsigned i){ return G(_thresholds[i][0], _thresholds[i][1]); };

            // Outgoing break-up momentum
            inline complex q(double m1, double m2)
            {
                return csqrt(Kallen(_s, m1*m1, m2*m2)) / csqrt(4.*_s);
            };
            inline complex q(unsigned i){ return q(_thresholds[i][0], _thresholds[i][1]); };

            // Product of momenta for the photoproduction process
            inline complex pq(unsigned i){ return _kinematics->initial_momentum(_s) * _q[i]; };
            
            // Product of momenta of the hadronic rescattering process
            inline complex q2(unsigned i, unsigned j){ return _q[i] * _q[j]; };

            std::vector<std::string> labels_2C = { J_label("n0"), J_label("n1"), J_label("a00"), J_label("a01"), J_label("a11") };
            std::vector<std::string> labels_3C = { J_label("n0"), J_label("n1"), J_label("n2"), J_label("a00"), J_label("a01"), J_label("a02"), J_label("a11"), J_label("a12"), J_label("a22") };
        };
    };
};

#endif