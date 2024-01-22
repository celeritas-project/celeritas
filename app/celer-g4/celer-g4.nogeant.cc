//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/celer-g4.nogeant.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <iostream>

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int, char* argv[])
{
    std::cerr << argv[0]
              << ": Geant4 is not enabled in this build of Celeritas"
              << std::endl;
    return EXIT_FAILURE;
}
