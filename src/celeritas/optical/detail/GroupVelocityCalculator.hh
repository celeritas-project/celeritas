//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/GroupVelocityCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

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
CELER_FUNCTION
GroupVelocityCalculator::GroupVelocityCalculator(MaterialView const& material)
    : r_index_calc_(material.make_refractive_index_calculator())
    , r_index_deriv_calc_(
          material.make_refractive_index_derivative_calculator())
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate group velocity for the given energy.
 */
CELER_FUNCTION real_type GroupVelocityCalculator::operator()(Energy energy) const
{
    real_type r_index = r_index_calc_(value_as<Energy>(energy));
    real_type r_index_deriv = r_index_deriv_calc_(value_as<Energy>(energy));

    real_type group_vel
        = constants::c_light
          / (r_index + value_as<Energy>(energy) * r_index_deriv);

    CELER_ENSURE(group_vel > 0);
    CELER_ENSURE(group_vel <= constants::c_light);

    return group_vel;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
