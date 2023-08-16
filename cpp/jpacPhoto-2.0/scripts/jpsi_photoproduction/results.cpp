// Script which imports the bootstrap error bounds and plots them alongside the 
// cross sections and data
//
// Reproduces figs. 2-4 of [1]
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
// REFERENCES:
// [1] 	arXiv:2305.01449 [hep-ph]
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

void results()
{
    using namespace jpacPhoto;
    using K_matrix         = analytic::K_matrix;

    plotter plotter;

    // Path to bootstrap files
    std::string path = "scripts/jpsi_photoproduction/bootstrap/";

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
    amplitude s_1C = new_amplitude<K_matrix>(kJpsi, 0, "1-channel S-wave");
    s_1C->set_option(K_matrix::kEffectiveRange);

    // Two-channel S-wave 
    amplitude s_2C = new_amplitude<K_matrix>(kJpsi, 0, higher, "2-channel S-wave");
    s_2C->set_option(K_matrix::kEffectiveRange);

    // Three-channel S-wave
    amplitude s_3C = new_amplitude<K_matrix>(kJpsi, 0, open_channels, "3-channel S-wave");

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
    sum_3CNR->set_id("Nonresonant (3C-NR)");

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

    // ------------------------------------------------------------------------------------
    // INTEGRATED PLOTS
    // ------------------------------------------------------------------------------------

    auto bs_int_1C   = import_transposed<8>(path + "1C/plot_xsec_gluex.txt");
    auto bs_int_2C   = import_transposed<8>(path + "2C/plot_xsec_gluex.txt");
    auto bs_int_3CNR = import_transposed<8>(path + "3C-NR/plot_xsec_gluex.txt");
    auto bs_int_3CR  = import_transposed<8>(path + "3C-R/plot_xsec_gluex.txt");

    // Plot all three curves with 95% confidence level in single plot
    plot pint1 = gluex::plot_integrated(plotter);
    pint1.set_curve_points(400);
    pint1.set_legend(0.5, 0.3);

    // Plot best-fit curve
    sum_1C->set_parameters(pars_1C);
    pint1.add_curve(sigma_Egam, sum_1C, {8,12});
    pint1.add_band( bs_int_1C[0], {bs_int_1C[4], bs_int_1C[5]});

    sum_2C->set_parameters(pars_2C);
    pint1.add_curve(sigma_Egam, sum_2C, {8,12});
    pint1.add_band( bs_int_2C[0], {bs_int_2C[4], bs_int_2C[5]});

    // Plot all three curves with 95% confidence level in single plot
    plot pint2 = gluex::plot_integrated(plotter);
    pint2.set_curve_points(200);
    pint2.set_legend(0.5, 0.3);
    pint2.color_offset(2);

    // Plot best-fit curve
    sum_3CNR->set_parameters(pars_3CNR);
    pint2.add_curve(sigma_Egam, sum_3CNR, {8,12});
    pint2.add_band( bs_int_3CNR[0], {bs_int_3CNR[4], bs_int_3CNR[5]});

    sum_3CR->set_parameters(pars_3CR);
    pint2.add_curve(sigma_Egam, sum_3CR, {8,12});
    pint2.add_band( bs_int_3CR[0], {bs_int_3CR[4], bs_int_3CR[5]});

    plotter.combine({2,1}, {pint1, pint2}, "gluex_int.pdf");

    //------------------------------------------------------------------------------------
    // DIFFERENTIAL PLOTS
    //------------------------------------------------------------------------------------
    
    // -----------------------------------------
    // GlueX
    
    // Grab data sets to access fixed Energy values
    std::vector<data_set> gluex_data = gluex::differential();

    std::vector<plot> gluex_difs_12C;
    std::vector<plot> gluex_difs_3C;
    for (int slice = 0; slice < 3; slice++)
    {
        plot pdif12 = gluex::plot_slice(plotter, slice);
        plot pdif3 = gluex::plot_slice(plotter, slice);
        pdif3.color_offset(2);
        
        double E_avg = gluex_data[slice]._avg_w;
        double W_avg = W_cm(E_avg);
        std::array<double,2> bounds = {-kJpsi->t_min(W_avg*W_avg), -kJpsi->t_max(W_avg*W_avg)};

        auto bs_dif_1C   = import_transposed<8>(path + "1C/plot_dsdt_gluex_"   +std::to_string(slice)+".txt");
        auto bs_dif_2C   = import_transposed<8>(path + "2C/plot_dsdt_gluex_"   +std::to_string(slice)+".txt");
        auto bs_dif_3CNR = import_transposed<8>(path + "3C-NR/plot_dsdt_gluex_"+std::to_string(slice)+".txt");
        auto bs_dif_3CR  = import_transposed<8>(path + "3C-R/plot_dsdt_gluex_" +std::to_string(slice)+".txt");

        sum_1C->set_parameters(pars_1C);
        pdif12.add_curve(dsigmadt_Egam, sum_1C, E_avg, bounds);
        pdif12.add_band( -bs_dif_1C[2], {bs_dif_1C[4], bs_dif_1C[5]});

        sum_2C->set_parameters(pars_2C);
        pdif12.add_curve(dsigmadt_Egam, sum_2C, E_avg, bounds);
        pdif12.add_band( -bs_dif_2C[2], {bs_dif_2C[4], bs_dif_2C[5]});

        sum_3CNR->set_parameters(pars_3CNR);
        pdif3.add_curve(dsigmadt_Egam, sum_3CNR, E_avg, bounds);
        pdif3.add_band( -bs_dif_3CNR[2], {bs_dif_3CNR[4], bs_dif_3CNR[5]});

        sum_3CR->set_parameters(pars_3CR);
        pdif3.add_curve(dsigmadt_Egam, sum_3CR, E_avg, bounds);
        pdif3.add_band( -bs_dif_3CR[2], {bs_dif_3CR[4], bs_dif_3CR[5]});
        
        if (slice != 0)
        {
            pdif12.set_legend(false); pdif3.set_legend(false); 
        };
        gluex_difs_12C.push_back(pdif12); gluex_difs_3C.push_back(pdif3);
    };
    plotter.stack(gluex_difs_12C, "gluex_dif_12C.pdf");
    plotter.stack(gluex_difs_3C,  "gluex_dif_3C.pdf");

    // -----------------------------------------
    // J/psi-007
    
    std::vector<plot> jpsi007_plots12, jpsi007_plots3;

    std::vector<data_set> jpsi007_data = jpsi007::all();
    for (int slice = 1; slice <= 12; slice++)
    {
        if (slice != 3 && slice != 5 && slice != 9) continue;
        double Eavg = jpsi007_data[slice-1]._avg_w;
        double Wavg = W_cm(Eavg);
        double tmin = -kJpsi->t_min(Wavg*Wavg);
        double tmax = -kJpsi->t_max(Wavg*Wavg);

        auto bs_dif_1C   = import_transposed<8>(path + "1C/plot_dsdt_007"    + std::to_string(slice-1)+".txt");
        auto bs_dif_2C   = import_transposed<8>(path + "2C/plot_dsdt_007"    + std::to_string(slice-1)+".txt");
        auto bs_dif_3CNR = import_transposed<8>(path + "3C-NR/plot_dsdt_007" + std::to_string(slice-1)+".txt");
        auto bs_dif_3CR  = import_transposed<8>(path + "3C-R/plot_dsdt_007"  + std::to_string(slice-1)+".txt");

        plot dif12 =  jpsi007::plot_slice(plotter, slice);

        plot dif3  =  jpsi007::plot_slice(plotter, slice);
        dif3.color_offset(2);

        amplitude to_plot;
        auto dsig_tp = [&](double tp)
        {
            double t = tp + tmin;
            return to_plot->differential_xsection(Wavg*Wavg, -t);
        };

        to_plot = sum_1C;
        to_plot->set_parameters(pars_1C);
        dif12.add_curve({0, tmax-tmin}, dsig_tp, to_plot->id());
        dif12.add_band( -bs_dif_1C[2]-tmin, {bs_dif_1C[4], bs_dif_1C[5]});

        to_plot = sum_2C;
        to_plot->set_parameters(pars_2C);
        dif12.add_curve({0, tmax-tmin}, dsig_tp, to_plot->id());
        dif12.add_band( -bs_dif_2C[2]-tmin, {bs_dif_2C[4], bs_dif_2C[5]});

        to_plot = sum_3CNR;
        to_plot->set_parameters(pars_3CNR);
        dif3.add_curve({0, tmax-tmin}, dsig_tp, to_plot->id());
        dif3.add_band( -bs_dif_3CNR[2]-tmin, {bs_dif_3CNR[4], bs_dif_3CNR[5]});

        to_plot = sum_3CR;
        to_plot->set_parameters(pars_3CR);
        dif3.add_curve({0, tmax-tmin}, dsig_tp, to_plot->id());
        dif3.add_band( -bs_dif_3CR[2]-tmin, {bs_dif_3CR[4], bs_dif_3CR[5]});

        // Adjust plot range
        if (slice == 3) 
        {
            dif12.set_ranges({0,4.5}, {9E-3, 1});
            dif12.set_legend(0.5, 0.55);

            dif3.set_ranges({0,4.5}, {9E-3, 1});
            dif3.set_legend(0.5, 0.55);
        }
        if (slice == 5) 
        {
            dif12.set_ranges({0,4.5}, {5E-3, 0.9});
            dif12.set_legend(false);

            dif3.set_ranges({0,4.5}, {5E-3, 0.9});
            dif3.set_legend(false);
        } 
        if (slice == 9 )
        {
             dif12.set_ranges({0,4.5}, {1E-3, 2});
             dif12.set_legend(0.3,0.35);
             dif12.set_legend(false);

             dif3.set_ranges({0,4.5}, {1E-3, 2});
             dif3.set_legend(0.3,0.35);
             dif3.set_legend(false);
        }

        jpsi007_plots12.push_back(dif12);
        jpsi007_plots3.push_back(dif3);
    };

    plotter.stack(jpsi007_plots12, "jpsi007_dif_12C.pdf");
    plotter.stack(jpsi007_plots3,  "jpsi007_dif_3C.pdf");
};