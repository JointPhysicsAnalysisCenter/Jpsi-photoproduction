// Script for extracting dynamical quantities form the fit results of the S-wave 
// K-matrices in J/psi photo production
//
// OUTPUT: zeta_th, R_VMD, and a_psip printed in table on cmdline
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#include "constants.hpp"
#include "kinematics.hpp"
#include "plotter.hpp"
#include "analytic/K_matrix.hpp"
#include "gluex/data.hpp"
#include "gluex/plots.hpp"

#include <cstring>
#include <iostream>
#include <iomanip>

void dynamics()
{
    using namespace jpacPhoto;
    using K_matrix         = analytic::K_matrix;

    // J/psi proton final
    kinematics kJpsi = new_kinematics(M_JPSI, M_PROTON);
    kJpsi->set_meson_JP( {1, -1} );
    
    // Threshold beam energy
    double Wth = kJpsi->Wth();
    double Eth = E_beam(Wth);

    // Open charm channels
    std::array<double,2> lower  = {M_D,     M_LAMBDAC};
    std::array<double,2> higher = {M_DSTAR, M_LAMBDAC};
    std::array<std::array<double,2>,2> both = {lower, higher};

    // ---------------------------------------------------------------------------
    // One-Channel (Elastic J/psi p)
    // ---------------------------------------------------------------------------

    std::vector<double> pars_1c = {-0.063170013, -418.239953512256, 320.763003700887};
    double a1_1c = -133.771684286972;

    // Create it as a pointer to generic amplitude
    amplitude x1c = new_amplitude<K_matrix>(kJpsi, 0, "1C");
    x1c->set_option(K_matrix::kEffectiveRange);
    x1c->set_parameters(pars_1c);
    
    // ---------------------------------------------------------------------------
    // Two-Channel (J/psi p and D* Lambdac)
    // ---------------------------------------------------------------------------

    std::vector<double> pars_2c = { 0.10122243, 3.2136095,
                                    -219.68553, 4.9978015, 47.09921,
                                    -181.30566, -145.67629};                // b's
    double a1_2c = -43.997429;

    // Create it as a pointer to generic amplitude
    amplitude x2c = new_amplitude<K_matrix>(kJpsi, 0, higher, "2C");
    x2c->set_option(K_matrix::kEffectiveRange);
    x2c->set_parameters(pars_2c);

    // ---------------------------------------------------------------------------
    // Three-Channel (J/psi p and both open chamrs)
    // ---------------------------------------------------------------------------

    std::vector<double> pars_3cNR = {   0.10543696, -0.10348633, -0.088686069,
                                        -258.12161, 168.24389, -132.60186,
                                        -135.93734, 235.48478,  93.983194 };
    double a1_3cNR = -61.240511;

    std::vector<double> pars_3cR = {    0.0087745655, 9.7952208, -0.011753894,
                                        -86.750775, -1.3425496, -88.96598,
                                        224.24897, 0.081102111, -294.93964 };
    double a1_3cR = -87.797765;

    // Create it as a pointer to generic amplitude
    amplitude x3cNR = new_amplitude<K_matrix>(kJpsi, 0, both, "3C-NR");
    x3cNR->set_parameters(pars_3cNR);

    amplitude x3cR = new_amplitude<K_matrix>(kJpsi, 0, both, "3C-R");
    x3cR->set_parameters(pars_3cR);

    // ---------------------------------------------------------------------------
    // Results
    // ---------------------------------------------------------------------------

    std::array<amplitude,4> amps    = {x1c, x2c, x3cNR, x3cR};
    std::array<double,4>    a_pwave = {a1_1c, a1_2c, a1_3cNR, a1_3cR};
    
    line(); 
    divider(4);
    print("", "zeta_th", "R_VMD x 10^2", "a_psip [fm]");
    divider(4);
    for (int i = 0; i < 4; i++)
    {
        // Downcast to K_matrix to access non-virtual functions
        auto amp = std::dynamic_pointer_cast<K_matrix>( amps[i] );

        // Calcuate elastic amplitude at threshold
        double a  = amp->scattering_length(); 

        // Print in a little table
        print(amp->id(), amp->zeta(), amp->R_VMD()*100, a);
    }
    divider(4); line();
};
