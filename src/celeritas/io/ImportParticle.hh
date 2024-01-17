//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportParticle.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store particle data.
 */
struct ImportParticle
{
    std::string name;
    int pdg{0};
    double mass{0};  //!< [MeV]
    double charge{0};  //!< [Multiple of electron charge value]
    double spin{0};  //!< [Multiple of hbar]
    double lifetime{0};  //!< [time]
    bool is_stable{false};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
