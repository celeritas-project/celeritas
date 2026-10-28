//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/GroupVelocityGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/inp/Grid.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 *Build group velocity grid using the rindex
 * grid and its interpolation calculator
 */
inp::Grid build_group_velocity_grid(
    inp::Grid const& refractive_index_grid,
    NonuniformGridCalculator refractive_index_calculator);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
