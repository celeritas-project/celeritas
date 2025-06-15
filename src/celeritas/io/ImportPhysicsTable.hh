//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportPhysicsTable.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas/inp/Grid.hh"

#include "ImportUnits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Imported physics table. Each table stores physics vectors for all materials.
 */
struct ImportPhysicsTable
{
    ImportUnits x_units{ImportUnits::unitless};
    ImportUnits y_units{ImportUnits::unitless};
    std::vector<inp::UniformGrid> grids;

    explicit operator bool() const { return !grids.empty(); }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
