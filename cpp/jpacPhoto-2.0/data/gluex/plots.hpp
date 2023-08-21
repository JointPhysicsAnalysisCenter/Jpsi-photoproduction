// Methods for quickly visualizing GlueX data and plotting theoretical curves
// compared to data points
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef GLUEX_PLOTS_HPP
#define GLUEX_PLOTS_HPP

#include "data_set.hpp"
#include "plotter.hpp"
#include "data.hpp"

namespace jpacPhoto
{
    namespace gluex
    {
        // Take in a plotter (just to prove colors have been initialized)
        // and produce the pot for the integrated cross_section
        inline plot plot_integrated(plotter& p)
        {
            data_set inte = gluex::integrated(); 

            plot pint = p.new_plot();
            pint.add_data( inte );
            pint.set_logscale(false, true);
            pint.set_legend(0.2, 0.65);
            pint.set_ranges({8, 12}, {1E-2, 10});
            pint.set_labels("#it{E}_{#gamma}  [GeV]", 
                            "#sigma(#gamma #it{p} #rightarrow #it{J}/#psi #it{p})  [nb]");

            return pint;
        };

        // Plot the differential 
        inline plot plot_slice(plotter& p, int i)
        {
            data_set slice = gluex::slice(i); 
            slice._id = "GlueX (2023)";
            
            // Grab the position of upper edge of last t-bin
            double tmax = slice._t.back() + slice._terr[1].back();

            plot pdif = p.new_plot();
            pdif.add_data(slice);
            pdif.set_logscale(false, true);
            pdif.set_legend(0.6, 0.4+(i!=0)*0.23);
            pdif.set_ranges({0, 10}, {3E-4, 6});
            if (i == 0) pdif.set_ranges({0, 10}, {7E-3, 1});
            if (i == 1) pdif.set_ranges({0, 10}, {2E-3, 3});
            pdif.add_header("#it{E}_{#gamma} =", slice._avg_w, "GeV");
            pdif.set_labels("#minus#it{t}  [GeV^{2}]", "#it{d}#sigma/#it{dt}  [nb / GeV^{2}]");

            return pdif;
        };

        // Plot the differential 
        inline plot plot_forward_slice(plotter& p, int i)
        {
            data_set slice = gluex::forward_slice(i); 
            slice._id = "GlueX"; 

            // Grab the position of upper edge of last t-bin
            double tmax = slice._t.back() + slice._terr[1].back();

            plot pdif = p.new_plot();
            pdif.add_data(slice);
            pdif.set_logscale(false, true);
            pdif.set_legend(0.55, 0.65);
            pdif.set_ranges({0, tmax + 0.2}, {1E-3, 6});
            pdif.add_header("#it{E}_{#gamma} =", slice._avg_w, "GeV");
            pdif.set_labels("#minus#it{t}  [GeV^{2}]", "#it{d}#sigma/#it{dt}  [nb / GeV^{2}]");

            return pdif;
        };
    };
};

#endif