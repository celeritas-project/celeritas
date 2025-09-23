//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/UnifiedReflectionCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/math/ArrayOperators.hh"
#include "celeritas/optical/Types.hh"

#include "SurfaceInteraction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Calculate geometric reflection of an incident vector about a normal.
 */
inline CELER_FUNCTION Real3 geometric_reflection(Real3 const& dir,
                                                 Real3 const& normal)
{
    return dir - 2 * dot_product(dir, normal) * normal;
}

//---------------------------------------------------------------------------//
/*!
 * Calculator for UNIFIED reflection model.
 *
 * The model specifies 4 different reflection modes:
 *  1. Specular spike: geometric reflection about global normal
 *  2. Specular lobe: geometric reflection about facet normal
 *  3. Back-scattering: reversed photon direction and polarization
 *  4. Diffuse Lambertian: reflection following Lambert's cosine law
 *
 * Only one reflection mode is selected based on the provided list of
 * probabilities.
 */
class UnifiedReflectionCalculator
{
  public:
    // Supported reflection modes in UNIFIED model
    enum class Modes
    {
        specular_spike,
        specular_lobe,
        back_scattering,
        diffuse_lambertian,
        size_
    };

    //!@{
    //! \name Type aliases
    using ModeProbs = EnumArray<Modes, real_type>;
    //!@}

  public:
    // Construct from mode probabilities, photon, and surface data
    explicit inline CELER_FUNCTION
    UnifiedReflectionCalculator(ModeProbs const& probs,
                                PhotonPhasor const& inc_photon,
                                Real3 const& global_normal,
                                Real3 const& facet_normal);

    // Sample reflection mode and calculate reflected phasor
    template<class Engine>
    inline CELER_FUNCTION PhotonPhasor operator()(Engine& rng) const;

    // Calculate specular spike reflection
    inline CELER_FUNCTION PhotonPhasor specular_spike() const;

    // Calculate specular lobe reflection
    inline CELER_FUNCTION PhotonPhasor specular_lobe() const;

    // Calculate back-scattering reflection
    inline CELER_FUNCTION PhotonPhasor back_scattering() const;

    // Sample diffuse Lambertian reflection
    template<class Engine>
    inline CELER_FUNCTION PhotonPhasor lambertian_reflection(Engine& rng) const;

  private:
    ModeProbs const& mode_probs_;
    PhotonPhasor const& inc_photon_;
    Real3 const& global_normal_;
    Real3 const& facet_normal_;

    // Calculate specular reflection about the given normal
    inline CELER_FUNCTION PhotonPhasor
    specular_reflection(Real3 const& normal) const;
};

//---------------------------------------------------------------------------//
/*!
 * Construct calculator from probabilities, photon, and surface data.
 */
CELER_FUNCTION UnifiedReflectionCalculator::UnifiedReflectionCalculator(
    ModeProbs const& probs,
    PhotonPhasor const& inc_photon,
    Real3 const& global_normal,
    Real3 const& facet_normal)
    : mode_probs_(probs)
    , inc_photon_(inc_photon)
    , global_normal_(global_normal)
    , facet_normal_(facet_normal)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample reflection mode from probabilities and calculate reflection.
 */
template<class Engine>
CELER_FUNCTION PhotonPhasor UnifiedReflectionCalculator::operator()(Engine&) const
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate specular spike reflection.
 *
 * This is geometric reflection about the global normal.
 */
CELER_FUNCTION PhotonPhasor UnifiedReflectionCalculator::specular_spike() const
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate specular lobe reflection.
 *
 * This is geometric reflection about the facet normal.
 */
CELER_FUNCTION PhotonPhasor UnifiedReflectionCalculator::specular_lobe() const
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate back-scattering reflection.
 *
 * The photon direction and polarization are reversed.
 */
CELER_FUNCTION PhotonPhasor UnifiedReflectionCalculator::back_scattering() const
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Sample diffuse Lambertian reflection.
 *
 * Ideal diffuse reflection following Lambert's cosine law.
 */
template<class Engine>
CELER_FUNCTION PhotonPhasor
UnifiedReflectionCalculator::lambertian_reflection(Engine&) const
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to calculate geometric reflection about a given normal.
 */
CELER_FUNCTION PhotonPhasor
UnifiedReflectionCalculator::specular_reflection(Real3 const&) const
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
