//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportParticle.hh
//! Store particle data.
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store particle data.
 *
 * ImportParticle objects are stored into a ROOT file by the app/geant-exporter
 * and retrieved by the RootImporter class.
 */
struct ImportParticle
{
    std::string name;
    int         pdg;
    real_type   mass; // [MeV]
    real_type   charge;
    real_type   spin;
    real_type   lifetime; // [s]
    bool        is_stable;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
