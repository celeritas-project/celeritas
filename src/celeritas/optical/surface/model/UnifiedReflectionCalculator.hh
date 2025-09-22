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

inline CELER_FUNCTION PhotonState
geometric_reflection(PhotonState const& inc_photon, Real3 const& normal)
{
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
    template<class T>
    using ModeArray = EnumArray<UnifiedReflectionModes, T>;
    //!@}

  public:
    template<class Engine>
    inline CELER_FUNCTION PhotonState operator()(Engine& rng) const;

  private:
    ModeArray<real_type> const& mode_probs_;
    Real3 const& global_normal_;
    Real3 const& facet_normal_;
    PhotonState const& inc_photon_;

    inline CELER_FUNCTION PhotonState
    geometric_reflection(Real3 const& normal) const;

    template<class Engine>
    inline CELER_FUNCTION PhotonState lambertian_reflection(Engine& rng) const;

    inline CELER_FUNCTION PhotonState back_scattering() const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
