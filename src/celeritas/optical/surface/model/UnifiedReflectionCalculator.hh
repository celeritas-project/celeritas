//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/UnifiedReflectionCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{

inline CELER_FUNCTION Real3 geometric_reflection(Real3 const& dir,
                                                 Real3 const& normal)
{
    return dir - 2 * dot_product(dir, normal) * normal;
}

enum class UnifiedReflectionModes
{
    specular_spike,
    specular_lobe,
    back_scattering,
    diffuse_lambertian,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    UnifiedReflectionCalculator ...;
   \endcode
 */
class UnifiedReflectionCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using ModeProbs = EnumArray<UnifiedReflectionModes, real_type>;
    //!@}

  public:
    UnifiedReflectionCalculator(ModeProbs const& probs,
                                PhotonPhasor const& inc_photon,
                                Real3 const& global_normal,
                                Real3 const& facet_normal);

    template<class Engine>
    inline CELER_FUNCTION PhotonPhasor operator()(Engine& rng) const;

  private:
    ModeArray<real_type> const& mode_probs_;
    PhotonPhasor const& inc_photon_;
    Real3 const& global_normal_;
    Real3 const& facet_normal_;

    inline CELER_FUNCTION PhotonPhasor
    specular_reflection(Real3 const& normal) const;

    template<class Engine>
    inline CELER_FUNCTION PhotonPhasor lambertian_reflection(Engine& rng) const;

    inline CELER_FUNCTION PhotonPhasor back_scattering() const;
};

UnifiedReflectionCalculator(ModeProbs const& probs,
                            PhotonPhasor const& inc_photon,
                            Real3 const& global_normal,
                            Real3 const& facet_normal)
    : mode_probs_(probs)
    , inc_photon_(inc_photon)
    , global_normal_(global_normal)
    , facet_normal_(facet_normal)
{
}

template<class Engine>
CELER_FUNCTION PhotonPhasor UnifiedReflectionCalculator::operator()(Engine&) const
{
    return {};
}

CELER_FUNCTION PhotonPhasor
UnifiedReflectionCalculator::specular_reflection(Real3 const&) const
{
    return {};
}

template<class Engine>
CELER_FUNCTION PhotonPhasor
UnifiedReflectionCalculator::lambertian_reflection(Engine&) const
{
    return {};
}

CELER_FUNCTION PhotonPhasor UnifiedReflectionCalculator::back_scattering() const
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
