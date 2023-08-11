//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3RootPrimary.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>
#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Flattened data stored/read to/from a ROOT file via \c HepMC3RootWriter and
 * \c HepMC3RootReader .
 */
struct HepMC3RootPrimary
{
    std::size_t event_id;
    int particle;
    double energy;
    double time;
    std::array<double, 3> pos;
    std::array<double, 3> dir;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
