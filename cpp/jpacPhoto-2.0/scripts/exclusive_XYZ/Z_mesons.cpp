// Photoproduction cross-sections for axial vector Z-mesons in charm and bottom sectors
// Reproduces fig. 2 of [1] in pdf form.
//
// OUTPUT: Z_mesons.pdf
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

#include "analytic/pseudoscalar_exchange.hpp"
#include "covariant/pseudoscalar_exchange.hpp"
#include "regge/pseudoscalar_exchange.hpp"

void Z_mesons()
{
    using namespace jpacPhoto;

    // ---------------------------------------------------------------------------
    // Couplings and constants
    // ---------------------------------------------------------------------------

    // Bottom vertex coupling (pi - nucleon - nucleon)
    double g_piNN = sqrt(2) * sqrt(4*PI*13.81); 

    // Cutoff for exponential form factor
    double lambda_pi = .900;  // MeV 

    // Pion trajectory parameters
    double slope = 0.7; // GeV-2
    double inter = - M2_PION * slope;

    // Zc(3900) couplings 
    double gc_jpsi  = 1.91; // psi coupling before VMD scaling
    double gc_gamma = E * F_JPSI * gc_jpsi / M_JPSI;

    // Zb(10610) couplings
    double gb_upsilon1S = 0.49, gb_upsilon2S = 3.30, gb_upsilon3S = 9.22;
    double gb_gamma = E * (F_UPSILON1S * gb_upsilon1S / M_UPSILON1S 
                         + F_UPSILON2S * gb_upsilon2S / M_UPSILON2S
                         + F_UPSILON3S * gb_upsilon3S / M_UPSILON3S);  

    // Zb'(10650) couplings
    double gbp_upsilon1S = 0.21, gbp_upsilon2S = 1.47, gbp_upsilon3S = 4.8;
    double gbp_gamma = E * (F_UPSILON1S * gbp_upsilon1S / M_UPSILON1S 
                         +  F_UPSILON2S * gbp_upsilon2S / M_UPSILON2S
                         +  F_UPSILON3S * gbp_upsilon3S / M_UPSILON3S);  

    // ---------------------------------------------------------------------------
    // Kinematics
    // ---------------------------------------------------------------------------

    kinematics kZc  = new_kinematics( M_ZC3900);
    kinematics kZb  = new_kinematics(M_ZB10610);
    kinematics kZbp = new_kinematics(M_ZB10650);

    kZc->set_meson_JP( AXIALVECTOR);
    kZb->set_meson_JP( AXIALVECTOR);
    kZbp->set_meson_JP(AXIALVECTOR);

    // ---------------------------------------------------------------------------
    // Low Energy amplitudes (analytic)
    // ---------------------------------------------------------------------------

    // Zc(3900)
    amplitude   Zc = new_amplitude<analytic::pseudoscalar_exchange>(kZc,   M_PION, "#it{Z}_{c}(3900)");
    Zc->set_parameters({gc_gamma, g_piNN, lambda_pi});

    // Zb(10610)
    amplitude   Zb = new_amplitude<analytic::pseudoscalar_exchange>(kZb,   M_PION, "#it{Z}_{b}(10610)");
    Zb->set_parameters({gb_gamma, g_piNN, lambda_pi});

    // Zb'(10650)
    amplitude   Zbp = new_amplitude<analytic::pseudoscalar_exchange>(kZbp, M_PION, "#it{Z}_{b}'(10650)");
    Zbp->set_parameters({gbp_gamma, g_piNN, lambda_pi});

    // ---------------------------------------------------------------------------
    // High Energy amplitudes (regge)
    // ---------------------------------------------------------------------------
    
    // Zc(3900)
    amplitude   ZcR = new_amplitude<regge::pseudoscalar_exchange>(kZc, "#it{Z}_{c}(3900)");
    ZcR->set_parameters({gc_gamma, g_piNN, lambda_pi, inter, slope});

    // Zb(10610)
    amplitude   ZbR = new_amplitude<regge::pseudoscalar_exchange>(kZb, "#it{Z}_{b}(10610)");
    ZbR->set_parameters({gb_gamma, g_piNN, lambda_pi, inter, slope});

    // Zb'(10650)
    amplitude   ZbpR = new_amplitude<regge::pseudoscalar_exchange>(kZbp, "#it{Z}_{b}'(10650)");
    ZbpR->set_parameters({gbp_gamma, g_piNN, lambda_pi, inter, slope});

    // ---------------------------------------------------------------------------
    // Plot results
    // ---------------------------------------------------------------------------

    plotter plotter;

    // Low-energy plot
    plot p_low = plotter.new_plot();
    p_low.set_curve_points(100);
    p_low.set_logscale(false, true);
    p_low.set_legend(0.7, 0.65);
    p_low.set_ranges({4, 20}, {2E-2, 2E2});
    p_low.set_labels( "#it{W_{#gammap}}  [GeV]", "#it{#sigma(#gamma p #rightarrow Z n)}  [nb]");
    p_low.add_curve(sigma_w, {Zc, Zb, Zbp}, {4, 20});

    // High-energy plot
    plot p_high = plotter.new_plot();
    p_high.set_curve_points(100);
    p_high.set_logscale(false, true);
    p_high.set_legend(0.3, 0.2);
    p_high.set_ranges({20, 70}, {1E-4, 2});
    p_high.set_labels( "#it{W_{#gammap}}  [GeV]", "#it{#sigma(#gamma p #rightarrow Z n)}  [nb]");
    p_high.add_curve(sigma_w, {ZcR, ZbR, ZbpR}, {20, 70});

    // Combine these into a single plot 
    plotter.combine({2,1}, {p_low, p_high}, "Z_mesons.pdf");
};