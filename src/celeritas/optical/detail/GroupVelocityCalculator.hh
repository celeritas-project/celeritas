//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/GroupVelocityCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/optical/MaterialView.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the group velocity of an optical photon based on the refractive
 * index.
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
    NonuniformGridCalculator r_index_calc_;
    NonuniformGridCalculator r_index_deriv_calc_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from a material view.
 */
CELER_FUNCTION GroupVelocityCalculator::GroupVelocityCalculator(
    MaterialView const& material)
    : r_index_calc_(material.make_refractive_index_calculator())
    , r_index_deriv_calc_(
          material.make_refractive_index_derivative_calculator())
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate group velocity for the given energy.
 */
CELER_FUNCTION real_type GroupVelocityCalculator::operator()(
    Energy energy) const
{
    // Clamp photon energy to the refractive-index grid endpoints
    real_type const bounded_energy = clamp(value_as<Energy>(energy),
                                           r_index_calc_.grid().front(),
                                           r_index_calc_.grid().back());

    real_type r_index = r_index_calc_(bounded_energy);
    real_type r_index_deriv = r_index_deriv_calc_(bounded_energy);

    // Group index n_g = n + E dn/dE. In anomalous-dispersion regions
    // (decreasing refractive index) the classical group velocity concept
    // breaks down and the naive expression can exceed the speed of light or
    // change sign: clamp the group index to unity.
    real_type group_index = celeritas::max(
        r_index + bounded_energy * r_index_deriv, real_type(1));

    real_type group_vel = constants::c_light / group_index;

    CELER_ENSURE(group_vel > 0);
    CELER_ENSURE(group_vel <= constants::c_light);

    return group_vel;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
