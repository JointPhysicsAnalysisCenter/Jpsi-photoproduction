// Script which imports the bootstrap error bounds and plots them alongside the 
// cross sections and data
//
// Reproduces figs. 2-4 
//
// OUTPUT: integrated.pdf 
//         gluex_diff_12C.pdf,   gluex_diff_3C.pdf
//         jpsi007_diff_12C.pdf, jpis007_diff_3C.pdf
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#include "constants.hpp"
#include "plotter.hpp"
#include "analytic/K_matrix.hpp"

#include "gluex/data.hpp"
#include "gluex/plots.hpp"
#include "jpsi007/data.hpp"
#include "jpsi007/plots.hpp"
#include "elementwise.hpp"

#include <cstring>
#include <iostream>
#include <iomanip>

void radius()
{
    using namespace jpacPhoto;
    using K_matrix         = analytic::K_matrix;
    using complex = std::complex<double>;

    plotter plotter;

    // Path to bootstrap files
    std::string path = "tests/jpsi/bootstrap/";

    // ---------------------------------------------------------------------------
    // Amplitude setup
    // ---------------------------------------------------------------------------

    // J/psi proton final
    kinematics kJpsi = new_kinematics(M_JPSI, M_PROTON);
    kJpsi->set_meson_JP( {1, -1} );

    // Or with two-channels coupled
    std::array<double,2> lower  = {M_D,     M_LAMBDAC};
    std::array<double,2> higher = {M_DSTAR, M_LAMBDAC};
    std::array<std::array<double,2>,2> open_channels = {lower, higher};

    // Single-channel S-wave
    amplitude s_1C = new_amplitude<K_matrix>(kJpsi, 0, "S-wave");
    s_1C->set_option(K_matrix::kEffectiveRange);

    // Two-channel S-wave 
    amplitude s_2C = new_amplitude<K_matrix>(kJpsi, 0, higher, "S-wave");
    s_2C->set_option(K_matrix::kEffectiveRange);

    // Three-channel S-wave
    amplitude s_3C = new_amplitude<K_matrix>(kJpsi, 0, open_channels, "S-wave");

    // The rest of the waves are single channel and 
    amplitude p = new_amplitude<K_matrix>(kJpsi, 1, "P-wave");
    amplitude d = new_amplitude<K_matrix>(kJpsi, 2, "D-wave");
    amplitude f = new_amplitude<K_matrix>(kJpsi, 3, "F-wave");

    // Assemble together
    amplitude sum_1C   = s_1C + p + d + f;
    sum_1C->set_id("Single channel (1C)");
    
    amplitude sum_2C   = s_2C + p + d + f;
    sum_2C->set_id("Two channels (2C)");

    amplitude sum_3CNR = s_3C + p + d + f;    
    sum_3CNR->set_id("Non-resonant (3C-NR)");

    amplitude sum_3CR  = s_3C + p + d + f;    
    sum_3CR->set_id("Resonant (3C-R)");

    // ------------------------------------------------------------------------------------
    // Parameters for each fit 

    // 1C
    std::vector<double> pars_1C = {
        0.063170013, -418.239953512256, 320.763003700887,
        0.018325241, -133.771684286972, 
        0.0030819158, -36.3240467916978, 
        0.0008141322, -25.9098777574191 };
    
    // 2C 
    std::vector<double> pars_2C = {
        0.10122243,    3.2136095, 
        -219.68553,    4.9978015, 47.09921,
        -181.30566,    -145.67629,
        0.014623196,   -43.997429,
        0.0030291014,  -2.3363761,
        0.00068965326, -6.0145674 };

    // 3C 
    std::vector<double> pars_3CNR = {
        0.10543696, -0.10348633, -0.088686069,
        -258.12161, 168.24389, -132.60186,
        -135.93734, 235.48478, 93.983194,
        0.016132013, -61.240511,
        0.0036324854, -4.7695843,
        0.00051931431, 3.1371592 };

    std::vector<double> pars_3CR = {
        0.0087745655, 9.7952208, -0.011753894, 
        -86.750775, -1.3425496, -88.96598,
        224.24897, 0.081102111, -294.93964,
        0.014023692, -87.797765,
        0.0036513428, -16.554736,
        0.00066429605, -10.174877 };

    print("J", "R [GeV^-1]", "R [fm]");

    auto R = [&](double s, auto wave, double f0)
    {
        auto pw = std::dynamic_pointer_cast<K_matrix>(wave);
        int J = pw->J();
        complex F = pw->partial_wave(s);

        double pq = kJpsi->initial_momentum(s) * kJpsi->final_momentum(s);
        double R2 =  pow(abs(F)/f0, 1./J) / pq;

        return sqrt(R2);
    };

    auto table = [&](double Eg, amplitude sum)
    {
        double W  = W_cm(Eg);
        double s  = W*W;

        auto swave = std::dynamic_pointer_cast<K_matrix>(extract_subamplitudes(sum)[0]);
        double f0 = std::abs(swave->partial_wave(s));
        print(sum->id());
        for (auto amp : extract_subamplitudes(sum))
        {
           auto pw = std::dynamic_pointer_cast<K_matrix>(amp);
            if (pw->J() == 0) continue;
           double r  = R(s, amp, f0);
           print(pw->J(), r, r/5.068); 
        }
    };

    divider();
    centered("E_GAMMA = 12 GeV");
    double Egam = 12.;
    sum_1C->set_parameters(pars_1C);
    table(Egam, sum_1C);
    sum_2C->set_parameters(pars_2C);
    table(Egam, sum_2C);
    sum_3CNR->set_parameters(pars_3CNR);
    table(Egam, sum_3CNR);
    sum_3CR->set_parameters(pars_3CR);
    table(Egam, sum_3CR);

    line();
    divider();
    Egam = E_beam(M_JPSI + M_PROTON) + EPS;
    centered("E_GAMMA = E_THRESHOLD");
    sum_1C->set_parameters(pars_1C);
    table(Egam, sum_1C);
    sum_2C->set_parameters(pars_2C);
    table(Egam, sum_2C);
    sum_3CNR->set_parameters(pars_3CNR);
    table(Egam, sum_3CNR);
    sum_3CR->set_parameters(pars_3CR);
    table(Egam, sum_3CR);
    line();

};