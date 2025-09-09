//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/EnergyLossCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "SplineCalculator.hh"
#include "UniformLogGridCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * For now, energy loss calculation has the same behavior as cross sections.
 *
 * The return value is [MeV / len] but isn't wrapped with a Quantity.
 */
class EnergyLossCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

    // Construct from state-independent data
    inline CELER_FUNCTION
    EnergyLossCalculator(UniformGridRecord const& grid, Values const& reals);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    UniformGridRecord const& data_;
    Values const& reals_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data.
 */
CELER_FUNCTION
EnergyLossCalculator::EnergyLossCalculator(UniformGridRecord const& grid,
                                           Values const& reals)
    : data_(grid), reals_(reals)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section using linear or spline interpolation.
 */
CELER_FUNCTION real_type EnergyLossCalculator::operator()(Energy energy) const
{
    if (data_.spline_order != 1)
    {
        // Spline interpolation without continuous derivatives
        return SplineCalculator(data_, reals_)(energy);
    }
    // Linear or cubic spline interpolation
    return UniformLogGridCalculator(data_, reals_)(energy);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
