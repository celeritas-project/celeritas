//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/GroupVelocityCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/optical/MaterialView.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Interpolate the optical-photon group velocity.
 *
 * Group velocity is precomputed from refractive-index data during optical
 * material construction.
 */
class GroupVelocityCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from a material view
    inline CELER_FUNCTION GroupVelocityCalculator(MaterialView const& material);

    // Calculate group velocity for the given energy
    inline CELER_FUNCTION real_type operator()(Energy) const;

  private:
    NonuniformGridCalculator group_velocity_calc_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from a material view.
 */
CELER_FUNCTION GroupVelocityCalculator::GroupVelocityCalculator(
    MaterialView const& material)
    : group_velocity_calc_(material.make_group_velocity_calculator())
{
}

//---------------------------------------------------------------------------//
/*!
 * Interpolate the group velocity using the precomputed group-velocity grid.
 */
CELER_FUNCTION real_type GroupVelocityCalculator::operator()(
    Energy energy) const
{
    real_type const group_vel = group_velocity_calc_(value_as<Energy>(energy));
    CELER_ENSURE(group_vel > 0 && group_vel <= constants::c_light);
    return group_vel;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
