//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportParticle.hh
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
 * ImportParticle objects are stored into a ROOT file by app/geant-exporter.
 *
 * \sa ParticleParamsLoader
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
