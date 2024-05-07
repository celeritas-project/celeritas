//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/WentzelVIXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/em/data/WentzelVIMscData.hh"
#include "celeritas/mat/MaterialView.hh"

#include "WentzelTransportXsCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the total cross section for the Wentzel VI MSC model.
 *
 * \note This performs the same calculation as the Geant4 method
 * G4WentzelVIModel::ComputeTransportXSectionPerVolume.
 */
class WentzelVIXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using XsUnits = units::Native;  // [1/len]
    //!@}

  public:
    // Construct with particle, material, and precalculatad Wentzel data
    inline CELER_FUNCTION
    WentzelVIXsCalculator(ParticleTrackView const& particle,
                          MaterialView const& material,
                          NativeCRef<WentzelVIMscData> const& data,
                          NativeCRef<WentzelOKVIData> const& wentzel,
                          Energy cutoff);

    // Compute the total cross section for the given angle
    inline CELER_FUNCTION real_type operator()(real_type cos_theta) const;

  private:
    ParticleTrackView const& particle_;
    MaterialView const& material_;
    CoulombIds const& ids_;
    CoulombParameters const& params_;
    Energy cutoff_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared model and material data.
 */
CELER_FUNCTION
WentzelVIXsCalculator::WentzelVIXsCalculator(
    ParticleTrackView const& particle,
    MaterialView const& material,
    NativeCRef<WentzelVIMscData> const& data,
    NativeCRef<WentzelOKVIData> const& wentzel,
    Energy cutoff)
    : particle_(particle)
    , material_(material)
    , ids_(data.ids)
    , params_(wentzel.params)
    , cutoff_(cutoff)
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute the total cross section for the given angle.
 */
CELER_FUNCTION real_type
WentzelVIXsCalculator::operator()(real_type cos_theta) const
{
    real_type result = 0;

    for (auto elcomp_id : range(ElementComponentId(material_.num_elements())))
    {
        AtomicNumber z = material_.make_element_view(elcomp_id).atomic_number();
        WentzelHelper helper(particle_, material_, z, params_, ids_, cutoff_);

        real_type costheta_max = helper.costheta_max_nuclear();
        if (costheta_max < cos_theta)
        {
            result += material_.elements()[elcomp_id.get()].fraction
                      * (helper.calc_xs_nuclear(cos_theta, costheta_max)
                         + helper.calc_xs_electron(cos_theta, costheta_max));
        }
    }
    result *= material_.number_density();

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
