//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/celer-geo.nojson.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <iostream>

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int, char* argv[])
{
    std::cerr << argv[0] << ": JSON is not enabled in this build of Celeritas"
              << std::endl;
    return EXIT_FAILURE;
}
