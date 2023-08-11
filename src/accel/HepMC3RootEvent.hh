//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3RootEvent.hh
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
 *
 * TODO: move to std::array<double, 3> for position and direction if
 * vector<array<double, 3>> becomes available without the need of a dictionary.
 */
struct HepMC3RootEvent
{
    std::size_t event_id;
    std::vector<int> particle;  //!< PDG number
    std::vector<double> energy;
    std::vector<double> time;
    std::vector<double> pos_x;
    std::vector<double> pos_y;
    std::vector<double> pos_z;
    std::vector<double> dir_x;
    std::vector<double> dir_y;
    std::vector<double> dir_z;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
