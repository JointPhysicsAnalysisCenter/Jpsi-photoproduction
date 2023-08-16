// Methods for quickly parsing Gluex 2022 data files into the data_set format
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef GLUEX_DATA_HPP
#define GLUEX_DATA_HPP

#include "data_set.hpp"
#include "constants.hpp"
#include "kinematics.hpp"

namespace jpacPhoto
{
    namespace gluex
    {
        // Integrated cross-section data as a function of Egamma
        inline data_set integrated()
        {
            std::string id = "GlueX (2023)";

            auto raw  = import_data<6>("/data/gluex/gluex2022_int.tsv");
            int  N    = check<6>(raw, id);
        
            data_set wrapped;
            wrapped._id   = id;
            wrapped._N    = N;
            wrapped._lab  = true;
            wrapped._type = integrated_data;
            wrapped._add_to_legend = true;

            // energy values are lab frame energies 
            wrapped._w       = raw[1];
            wrapped._werr[0] = raw[1] - raw[4];
            wrapped._werr[1] = raw[5] - raw[1];

            // cross section is column 2
            // column 3 is the error
            // compute symmetric upper and lower error bars
            wrapped._obs       = raw[2];
            wrapped._obserr    = raw[3]; 

            return wrapped;
        };

        // Parses the three slices available for differential cross-section
        inline data_set slice(int sliceid)
        {
            
            std::string id;
            std::string filename;
            double Eavg;

            switch (sliceid)
            {
                case 0: 
                {
                    id = "GlueX (E = 8.93 GeV)";
                    filename = "/data/gluex/gluex2022_diff_E0893.tsv";
                    Eavg = 8.92877;
                    break;
                }
                case 1:
                {
                    id = "GlueX (E = 9.85 GeV)";
                    filename = "/data/gluex/gluex2022_diff_E0985.tsv";
                    Eavg = 9.8583;
                    break;
                }
                case 2:
                {
                    id = "GlueX (E = 10.82 GeV)";
                    filename = "/data/gluex/gluex2022_diff_E1082.tsv";
                    Eavg = 10.8205;
                    break;
                }
                default: return data_set();
            };
            

            auto raw  = import_data<7>(filename);
            int  N    = check<7>(raw, id);

            data_set wrapped;
            wrapped._id    = id;
            wrapped._type  = differential_data;
            wrapped._N     = N;
            wrapped._add_to_legend = true;

            // These are differential and thus we need to specify the avg energy 
            wrapped._lab   = true;
            wrapped._negt  = true;
            wrapped._avg_w = Eavg;

            wrapped._w         = raw[6]; 
            wrapped._t         = raw[1];
            wrapped._terr[0]   = raw[1] - raw[4];
            wrapped._terr[1]   = raw[5] - raw[1];
            wrapped._obs       = raw[2];
            wrapped._obserr    = raw[3];

            return wrapped;
        };

        // Produce a vector with all the 2022 data from GlueX
        // Indices correspond to:
        // [0] slice-0 differential
        // [1] slice-1 differential
        // [2] slice-2 differential
        // [3] integrated xsection
        inline std::vector<data_set> all()
        {
            std::vector<data_set> data;
            for (int i = 0; i < 3; i++)
            {
                data.push_back( gluex::slice(i) );
            };
            data.push_back( gluex::integrated() );

            return data;
        };

        // Output the data points from each differential slice that correspong to cosTheta > 0
        inline data_set forward_slice(int sliceid)
        {

            std::string id;
            std::string filename;
            double Eavg;

            switch (sliceid)
            {
                case 0: 
                {
                    id = "GlueX (E = 8.93 GeV)";
                    filename = "/data/gluex/gluex2022_diff_E0893.tsv";
                    Eavg = 8.92877;
                    break;
                }
                case 1:
                {
                    id = "GlueX (E = 9.85 GeV)";
                    filename = "/data/gluex/gluex2022_diff_E0985.tsv";
                    Eavg = 9.8583;
                    break;
                }
                case 2:
                {
                    id = "GlueX (E = 10.82 GeV)";
                    filename = "/data/gluex/gluex2022_diff_E1082.tsv";
                    Eavg = 10.8205;
                    break;
                }
                default: return data_set();
            };
            
            // Grab the entire slice 
            auto raw = import_data<7>(filename);
            int Ntot = check<7>(raw, "GlueX forward slice");

            // Number of data points after filtering
            int N = 0;
            std::array<std::vector<double>,7> reduced;

            // The filtering requires kinematics of jpsi p 
            kinematics kJpsi = new_kinematics(M_JPSI, M_PROTON);

            for (int i = 0; i < Ntot; i++)
            {
                double W   = W_cm(raw[6][i]);
                double t   = - raw[1][i];
                double cos = kJpsi->z_s(W*W, t);

                // Filter step
                if (cos < 0) continue;

                // Populate new arrays
                N++;
                for (int j = 0; j < 7; j++)
                {
                    reduced[j].push_back(raw[j][i]);
                };
            };

            data_set wrapped;
            wrapped._id    = id;
            wrapped._type  = differential_data;
            wrapped._N     = N;

            // These are differential and thus we need to specify the avg energy 
            wrapped._lab   = true;
            wrapped._negt  = true;
            wrapped._avg_w = Eavg;

            wrapped._w         = reduced[6]; 
            wrapped._t         = reduced[1];
            wrapped._terr[0]   = reduced[1] - reduced[4];
            wrapped._terr[1]   = reduced[5] - reduced[1];
            wrapped._obs       = reduced[2];
            wrapped._obserr    = reduced[3];

            return wrapped;
        };

        // Produce a vector with all the 2022 data from GlueX in the forward direction (costheta > 0)
        // Indices correspond to:
        // [0] slice-0 differential
        // [1] slice-1 differential
        // [2] slice-2 differential
        // [3] integrated xsection
        inline std::vector<data_set> forward_all()
        {
            std::vector<data_set> data;
            for (int i = 0; i < 3; i++)
            {
                data.push_back( gluex::forward_slice(i) );
            };
            data.push_back( gluex::integrated() );

            return data;
        };

        // Produce a vector with all the 2022 differential data data from GlueX
        // Indices correspond to:
        // [0] slice-0 differential
        // [1] slice-1 differential
        // [2] slice-2 differential
        inline std::vector<data_set> differential()
        {
            std::vector<data_set> data;
            for (int i = 0; i < 3; i++)
            {
                data.push_back( gluex::slice(i) );
            };
            return data;
        };
    };
};

#endif