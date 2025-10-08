//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/ReflectionFormCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "DielectricInteractionData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * A view into UNIFIED model grids to calculate reflection mode probabilities.
 */
class ReflectionFormCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using DataRef = NativeCRef<UnifiedReflectionData>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from data, surface, and energy
    explicit inline CELER_FUNCTION
    ReflectionFormCalculator(DataRef const&, SubModelId, Energy);

    // Calculate probability for a specific reflection mode
    inline CELER_FUNCTION real_type operator()(UnifiedReflectionMode) const;

  private:
    DataRef const& data_;
    SubModelId surface_;
    Energy energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from data, surface, and energy.
 */
CELER_FUNCTION
ReflectionFormCalculator::ReflectionFormCalculator(DataRef const& data,
                                                   SubModelId surface,
                                                   Energy energy)
    : data_(data), surface_(surface), energy_(energy)
{
    CELER_EXPECT(surface_ < data_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate probability for the given reflection mode.
 *
 * Only the specular spike, specular lobe, and back-scattering probabilities
 * are defined as grids in the data. The diffuse Lambertian mode is the
 * remaining probability.
 */
CELER_FUNCTION real_type
ReflectionFormCalculator::operator()(UnifiedReflectionMode mode) const
{
    NonuniformGridCalculator calc{data_.reflection_grids[mode][surface_],
                                  data_.reals};
    real_type result = calc(value_as<Energy>(energy));
    CELER_ENSURE(result >= 0 && result <= 1);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
