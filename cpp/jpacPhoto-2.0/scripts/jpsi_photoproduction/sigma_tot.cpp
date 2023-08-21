// Script which imports the bootstrap error bounds and plots them alongside the 
// cross sections and data
//
// Reproduces figs. 6
//
// OUTPUT: sigma_tot.pdf
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

void sigma_tot()
{
    using namespace jpacPhoto;
    using K_matrix         = analytic::K_matrix;

    plotter plotter;

    // Path to bootstrap files
    std::string path = "scripts/jpsi_photoproduction/bootstrap/";

    auto convert_to_W = [=](std::vector<double> egam)
    {
        std::vector<double> ws;
        for (auto e : egam)
        {
            ws.push_back(W_cm(e));
        }
        return ws;
    };

    // ------------------------------------------------------------------------------------
    // PLOT RESULTS
    // ------------------------------------------------------------------------------------

    auto bs_1C   = import_transposed<8>(path + "1C/plot_total.txt");
    auto bs_2C   = import_transposed<8>(path + "2C/plot_total.txt");
    auto bs_3CNR = import_transposed<8>(path + "3C-NR/plot_total.txt");
    auto bs_3CR  = import_transposed<8>(path + "3C-R/plot_total.txt");

    auto Ws = convert_to_W(bs_1C[0]);

    double Wth = M_JPSI + M_PROTON + EPS;

    plot p2 = plotter.new_plot();
    p2.set_logscale(false, true);
    p2.set_legend(0.5, 0.3);
    p2.set_ranges({Wth, 5}, {4E-2, 4E2});
    p2.set_labels("#sqrt{#it{s}}  [GeV]", "#sigma^{#it{J}/#psi#it{p}}_{tot}  [mb]");

    p2.add_curve( Ws, (bs_1C[4] + bs_1C[5])/2, "One channel (1C)" );
    p2.add_band(  Ws, {bs_1C[4], bs_1C[5]});

    p2.add_curve( Ws, (bs_2C[4] + bs_2C[5])/2, "Two channels (2C)" );
    p2.add_band(  Ws, {bs_2C[4], bs_2C[5]});

    p2.add_curve( Ws, (bs_3CNR[4] + bs_3CNR[5])/2, "Non-resonant (3C-NR)" );
    p2.add_band(  Ws, {bs_3CNR[4], bs_3CNR[5]});

    p2.add_curve( Ws, (bs_3CR[4] + bs_3CR[5])/2, "Resonant (3C-R)" );
    p2.add_band(  Ws, {bs_3CR[4], bs_3CR[5]});
    p2.save("sigma_tot.pdf");
};