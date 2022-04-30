//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
 * \sa ImportData
 */
struct ImportParticle
{
    std::string name;
    int         pdg;
    double      mass;     //!< [MeV]
    double      charge;   //!< [Multiple of electron charge]
    double      spin;     //!< [Multiple of hbar]
    double      lifetime; //!< [s]
    bool        is_stable;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
