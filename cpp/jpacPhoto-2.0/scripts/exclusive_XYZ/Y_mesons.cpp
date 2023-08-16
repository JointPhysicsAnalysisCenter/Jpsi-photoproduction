// Photoproduction cross-sections for vector charmonium and charmonium-like mesons
// Reproduces fig. 5 of [1] in pdf form.
//
// OUTPUT: Y_mesons.pdf
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------
// REFERENCES:
// [1] arXiv:2008.01001 [hep-ph]
// ---------------------------------------------------------------------------

#include "constants.hpp"
#include "kinematics.hpp"
#include "plotter.hpp"

#include "analytic/pomeron_exchange.hpp"
#include "covariant/pomeron_exchange.hpp"

void Y_mesons()
{
    using namespace jpacPhoto;

    // ---------------------------------------------------------------------------
    // Couplings and constants
    // ---------------------------------------------------------------------------

    // Pomeron trajectory (high energies)
    double alpha_0H = 1.15;
    double alpha_PH = 0.11;

    // Pomeron trajectory (low energies)
    double alpha_0L = 0.94;
    double alpha_PL = 0.36;

    // Slope parameter and normalization (high)
    double bH = 1.01;
    double AH = E * 0.16;

    // Slope parameter and normalization (low)
    double bL = 0.12;
    double AL = E * 0.38;

    // Scale parameters
    double R_Jpsi  = 1;
    double R_Psi2S = 0.55;
    double R_Y     = 0.84; 

    // ---------------------------------------------------------------------------
    // Kinematics
    // ---------------------------------------------------------------------------

    kinematics kJpsi  = new_kinematics( M_JPSI );
    kJpsi->set_meson_JP(VECTOR);

    kinematics kPsi2S = new_kinematics( M_PSI2S );
    kPsi2S->set_meson_JP(VECTOR);

    kinematics kY     = new_kinematics( M_Y4260 );
    kY->set_meson_JP(VECTOR);

    // ---------------------------------------------------------------------------
    // Low Energy amplitudes (Vector-like coupling)
    // ---------------------------------------------------------------------------
    
    // Jpsi
    amplitude   JpsiL = new_amplitude<covariant::pomeron_exchange>(kJpsi, "#it{J}/#psi");
    JpsiL->set_parameters({AL*R_Jpsi, bL, alpha_0L, alpha_PL});

    // Psi(2S)
    amplitude   Psi2SL= new_amplitude<covariant::pomeron_exchange>(kPsi2S, "#psi(2#it{S})");
    Psi2SL->set_parameters({AL*R_Psi2S, bL, alpha_0L, alpha_PL});

    // Y(4260)
    amplitude   YL = new_amplitude<covariant::pomeron_exchange>(kY, "#it{Y}(4260)");
    YL->set_parameters({AL*R_Y, bL, alpha_0L, alpha_PL});

    // ---------------------------------------------------------------------------
    // High Energy amplitudes (Helicity-conserving)
    // ---------------------------------------------------------------------------
    
    // Jpsi
    amplitude   JpsiH = new_amplitude<analytic::pomeron_exchange>(kJpsi, "#it{J}/#psi");
    JpsiH->set_parameters({AH*R_Jpsi, bH, alpha_0H, alpha_PH});

    // Psi(2S)
    amplitude   Psi2SH = new_amplitude<analytic::pomeron_exchange>(kPsi2S, "#psi(2#it{S})");
    Psi2SH->set_parameters({AH*R_Psi2S, bH, alpha_0H, alpha_PH});

    // Y(4260)
    amplitude   YH = new_amplitude<analytic::pomeron_exchange>(kY, "#it{Y}(4260)");
    YH->set_parameters({AH*R_Y, bH, alpha_0H, alpha_PH});

    // ---------------------------------------------------------------------------
    // Plot results
    // ---------------------------------------------------------------------------

    plotter plotter;

    // Low-energy plot
    plot p_low = plotter.new_plot();
    p_low.set_curve_points(100);
    p_low.set_legend(0.3, 0.73);
    p_low.set_ranges({4, 10}, {0, 15});
    p_low.set_labels( "#it{W_{#gammap}}  [GeV]", "#sigma(#gamma#it{p} #rightarrow #it{Yp})  [nb]");
    p_low.add_curve(sigma_w, {JpsiL, Psi2SL, YL}, {4, 10});

    // High-energy plot
    plot p_high = plotter.new_plot();
    p_high.set_curve_points(100);
    p_high.set_legend(0.3, 0.73);
    p_high.set_ranges({30, 100}, {0, 100});
    p_high.set_labels( "#it{W_{#gammap}}  [GeV]", "#sigma(#gamma#it{p} #rightarrow #it{Yp})  [nb]");
    p_high.add_curve(sigma_w, {JpsiH, Psi2SH, YH}, {30, 100});

    // Combine these into a single plot 
    plotter.combine({2,1}, {p_low, p_high}, "Y_mesons.pdf");
};