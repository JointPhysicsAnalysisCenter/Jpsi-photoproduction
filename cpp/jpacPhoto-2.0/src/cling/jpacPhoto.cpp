// Load script that links all the required libraries.
// Adapted from the installation of elSpectro
// [https://github.com/dglazier/elSpectro]
// by Derek Glazier and others
//
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC)
// Email:        dwinney@alumni.iu.edu
// -----------------------------------------------------------------------------

#include <iostream>
#include <TRint.h>
#include <TEnv.h>
#include <TString.h>
#include <TSystem.h>
#include <vector>

int main(int argc, char **argv)
{
    TString macroName;
    for(Int_t i = 0; i < argc; i++)
    {
        TString opt = argv[i];
        if((opt.Contains(".C")))   macroName = opt;
        if((opt.Contains(".cpp"))) macroName = opt;
    }

    // All of this is just to add "-l" to argv so that 
    // ROOT opens without a welcome message...
    std::vector<char*> new_argv(argv, argv + argc);
    new_argv.push_back((char *) "-l");
    new_argv.push_back(NULL);
    argv = new_argv.data();
    argc += 1;    

    TRint * app = new TRint( "jpacPhoto", &argc, argv);
    TString JPACPHOTO = gSystem->Getenv("JPACPHOTO");
    
    if (JPACPHOTO.Length() == 0) std::cout << "Environment variable JPACPHOTO not set!" << std::endl;
    
    app->ProcessLine(".x $JPACPHOTO/src/cling/Load.C");
    app->ProcessLine(Form(".x %s", macroName.Data()));
    app->Terminate(0);

    delete app;
    return 0;
};