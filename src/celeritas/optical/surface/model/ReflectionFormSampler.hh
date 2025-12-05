//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/ReflectionFormSampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/distribution/Selector.hh"

#include "DielectricInteractionData.hh"
#include "ReflectionFormCalculator.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Calculate probability for each reflection mode from UNIFIED model grids.
 */
class ReflectionModeSampler
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
    ReflectionModeSampler(DataRef const&, SubModelId, Energy);

    // Calculate probability for a specific reflection mode
    inline CELER_FUNCTION real_type operator()(ReflectionMode) const;

  private:
    DataRef const& data_;
    SubModelId surface_;
    Energy energy_;
};

//---------------------------------------------------------------------------//
/*!
 * Sample a reflection result based on UNIFIED model grid probabilities.
 */
class ReflectionFormSampler
{
  public:
    // Construct from a mode sampler and a reflection calculator
    inline CELER_FUNCTION
    ReflectionFormSampler(ReflectionModeSampler sample_mode,
                          ReflectionFormCalculator const& calc_reflection);

    // Sample a surface interaction
    template<class Engine>
    inline CELER_FUNCTION SurfaceInteraction operator()(Engine&) const;

  private:
    Selector<ReflectionModeSampler, ReflectionMode> sample_mode_;
    ReflectionFormCalculator const& calc_reflection_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from data, surface, and energy.
 */
CELER_FUNCTION
ReflectionModeSampler::ReflectionModeSampler(DataRef const& data,
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
ReflectionModeSampler::operator()(ReflectionMode mode) const
{
    NonuniformGridCalculator calc{data_.reflection_grids[mode][surface_],
                                  data_.reals};
    real_type result = calc(value_as<Energy>(energy_));
    CELER_ENSURE(result >= 0 && result <= 1);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a mode sampler and reflection calculator.
 */
CELER_FUNCTION ReflectionFormSampler::ReflectionFormSampler(
    ReflectionModeSampler sample_mode,
    ReflectionFormCalculator const& calc_reflection)
    : sample_mode_{make_unnormalized_selector(
          std::move(sample_mode), ReflectionMode::size_, real_type{1})}
    , calc_reflection_(calc_reflection)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample a surface interaction.
 */
template<class Engine>
inline CELER_FUNCTION SurfaceInteraction
ReflectionFormSampler::operator()(Engine& rng) const
{
    switch (sample_mode_(rng))
    {
        case ReflectionMode::specular_spike:
            return calc_reflection_.calc_specular_spike();
        case ReflectionMode::specular_lobe:
            return calc_reflection_.calc_specular_lobe();
        case ReflectionMode::backscatter:
            return calc_reflection_.calc_backscatter();
        case ReflectionMode::diffuse_lobe:
            // The other probabilities are allowed to sum to less than unity:
            // the remainder is diffuse
            return calc_reflection_.sample_lambertian_reflection(rng);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}
//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
