// Draws the total pion-nucleon cross-section using JPAC amplitudes & PDG parameterization
// Reproduces Fig. 3 of [1]
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------
// REFERENCES:
// [1] 	arXiv:2209.05882 [hep-ph]
// ------------------------------------------------------------------------------

#include "sigma_tot/PDG.hpp"
#include "sigma_tot/JPAC_piN.hpp"
#include "plotter.hpp"

using namespace jpacPhoto;

void sigmatot_piN()
{
    using namespace jpacPhoto;
    using complex = std::complex<double>;

    total_xsection PDG_pimp  = new_PDG_sigmatot(pimp);
    total_xsection JPAC_pimp = new_total_xsection<JPAC_piN>(-1);

    total_xsection PDG_pipp  = new_PDG_sigmatot(pipp);
    total_xsection JPAC_pipp = new_total_xsection<JPAC_piN>(+1);

    total_xsection to_plot;
    auto sig = [&](double w)
    {
        return to_plot->evaluate(w*w, M2_PION);
    };

    plotter plotter;
    plot p = plotter.new_plot();

    double Wth = M_PION + M_PROTON + 1E-3;
    to_plot = JPAC_pipp;
    p.add_curve({Wth, 2.5}, sig, "#pi^{#plus} #it{p}");
    to_plot = PDG_pipp;
    p.add_dashed({Wth,2.5}, sig);

    to_plot = JPAC_pimp;
    p.add_curve({Wth, 2.5}, sig, "#pi^{#minus} #it{p}");
    to_plot = PDG_pimp;
    p.add_dashed({Wth,2.5}, sig);

    p.set_ranges({1, 2.5}, {6, 400});
    p.set_logscale(false, true);
    p.set_labels("#it{W}_{#gamma#it{p}}  [GeV]", "#sigma_{tot}^{#pi#it{p}} [mb]");
    p.set_legend(0.6, 0.7);
    p.save("sigmatot.pdf");
};