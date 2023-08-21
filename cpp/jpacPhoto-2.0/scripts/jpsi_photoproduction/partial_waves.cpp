// Script to examine the breakdown of different L components to the total 
// cross-section from the 2-channel model compared to a pomeron exchange
// Reproduces Fig. 5 of [1]
//
// OUTPUT: jpsi007_results.pdf, gluex_results.pdf, paper_figure.pdf
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------
// REFERENCES:
// [1] 	arXiv:2305.01449 [hep-ph]
// ------------------------------------------------------------------------------


#include "constants.hpp"
#include "kinematics.hpp"
#include "partial_wave.hpp"
#include "plotter.hpp"

#include "analytic/K_matrix.hpp"
#include "gluex/data.hpp"
#include "gluex/plots.hpp"
#include "jpsi007/data.hpp"
#include "jpsi007/plots.hpp"

#include <cstring>
#include <iostream>
#include <iomanip>

void partial_waves()
{
    using namespace jpacPhoto;
    using K_matrix         = analytic::K_matrix;

    // ---------------------------------------------------------------------------
    //  Parameters
    // ---------------------------------------------------------------------------

    // Or with two-channels coupled
    std::array<double,2> lower  = {M_D,     M_LAMBDAC};
    std::array<double,2> higher = {M_DSTAR, M_LAMBDAC};
    std::array<std::array<double,2>,2> open_channels = {lower, higher};

    // 2C 
    std::vector<double> pars_3CNR = {
        0.10543696, -0.10348633, -0.088686069,
        -258.12161, 168.24389, -132.60186,
        -135.93734, 235.48478, 93.983194,
        0.016132013, -61.240511,
        0.0036324854, -4.7695843,
        0.00051931431, 3.1371592 };

    // ---------------------------------------------------------------------------
    // Amplitude setup
    // ---------------------------------------------------------------------------

    // J/psi proton final state this is shared by both amplitudes
    kinematics kJpsi = new_kinematics(M_JPSI, M_PROTON);
    kJpsi->set_meson_JP( {1, -1} );

    // K-MATRIX
    amplitude s = new_amplitude<K_matrix>(kJpsi, 0, open_channels, "S-wave");
    amplitude p = new_amplitude<K_matrix>(kJpsi, 1, "P-wave");
    amplitude d = new_amplitude<K_matrix>(kJpsi, 2, "D-wave");
    amplitude f = new_amplitude<K_matrix>(kJpsi, 3, "F-wave");

    amplitude sum_kmatrix = s + p + d + f;
    sum_kmatrix->set_id("Sum (3C-NR)");
    sum_kmatrix->set_parameters(pars_3CNR);

    std::vector<amplitude> amps = {sum_kmatrix, s, p, d, f};

    // ---------------------------------------------------------------------------
    // Plot the results
    // ---------------------------------------------------------------------------

    plotter plotter;

    // All GlueX 2022 data
    std::vector<data_set> gluex   = gluex::all();
    std::vector<data_set> jpsi007 = jpsi007::all(); // And all Jpsi-007 data

    // -----------------------------------------
    // GlueX 

    std::vector<plot> gluex_plots;

    // Grab each pre-set plot but add the theory curve with add_curve
    plot pint = gluex::plot_integrated(plotter);

    for (int i = 0; i < 5; i++)
    {
        pint.add_curve(jpacPhoto::sigma_Egam, amps[i], {8, 11.8});
    }
    gluex_plots.push_back(pint);

    // Do the same with the differential sets
    for (int i = 0; i <= 2; i++)
    {
        double Eavg = gluex[i]._avg_w;
        double Wavg = W_cm(Eavg);
        double tmin = -kJpsi->t_min(Wavg*Wavg);
        double tmax = -kJpsi->t_max(Wavg*Wavg); 

        plot dif = gluex::plot_slice(plotter, i);

        for (int j = 0; j < 5; j++)
        {
            dif.add_curve(jpacPhoto::dsigmadt_Egam, amps[j], Eavg, {tmin, tmax});
        }
        if (i == 0) dif.set_ranges({0,6}, {4E-3, 1});
        if (i == 1) dif.set_ranges({0,8.2}, {4E-3, 3});
        dif.set_legend(0.5, 0.6);
        gluex_plots.push_back(dif);
    };
    // // Print to file as a 2x2 grid
    plotter.combine({2,2}, gluex_plots, "gluex_results.pdf");
    plotter.combine({2,1}, {gluex_plots[0], gluex_plots[2]}, "paper_figure.pdf");
    // -----------------------------------------
    // J/psi-007
    
    std::vector<plot> jpsi007_plots;

    for (int i = 1; i <= 12; i++)
    {
        double Eavg = jpsi007[i-1]._avg_w;
        double Wavg = W_cm(Eavg);
        double tmin = -kJpsi->t_min(Wavg*Wavg);
        double tmax = -kJpsi->t_max(Wavg*Wavg);

        plot dif =  jpsi007::plot_slice(plotter, i);

        for (int i = 0; i < 5; i++)
        {
            auto dsig_tp = [&](double tp)
            {
                double t = tp + tmin;
                return amps[i]->differential_xsection(Wavg*Wavg, -t);
            };
            dif.add_curve({0, tmax-tmin}, dsig_tp, amps[i]->id());
        }
        jpsi007_plots.push_back(dif);
    };

    plotter.combine({4,3}, jpsi007_plots, "jpsi007_results.pdf");
};