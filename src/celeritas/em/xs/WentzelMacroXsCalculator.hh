//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/WentzelMacroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/data/WentzelOKVIData.hh"
#include "celeritas/em/data/WentzelVIMscData.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "WentzelHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the total cross section for the Wentzel VI MSC model.
 *
 * \note This performs the same calculation of the total cross section (\c
 * xtsec) as the Geant4 method
 * G4WentzelVIModel::ComputeTransportXSectionPerVolume.
 */
class WentzelMacroXsCalculator
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
    WentzelMacroXsCalculator(ParticleTrackView const& particle,
                             MaterialView const& material,
                             NativeCRef<WentzelVIMscData> const& data,
                             NativeCRef<WentzelOKVIData> const& wentzel,
                             Energy cutoff);

    // Compute the total cross section for the given angle
    inline CELER_FUNCTION real_type operator()(real_type cos_theta) const;

  private:
    ParticleTrackView const& particle_;
    MaterialView const& material_;
    NativeCRef<WentzelOKVIData> const& wentzel_;
    CoulombIds const& ids_;
    Energy cutoff_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared model and material data.
 */
CELER_FUNCTION
WentzelMacroXsCalculator::WentzelMacroXsCalculator(
    ParticleTrackView const& particle,
    MaterialView const& material,
    NativeCRef<WentzelVIMscData> const& data,
    NativeCRef<WentzelOKVIData> const& wentzel,
    Energy cutoff)
    : particle_(particle)
    , material_(material)
    , wentzel_(wentzel)
    , ids_(data.ids)
    , cutoff_(cutoff)
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute the total cross section for the given angle.
 */
CELER_FUNCTION real_type
WentzelMacroXsCalculator::operator()(real_type cos_theta) const
{
    real_type result = 0;

    for (auto elcomp_id : range(ElementComponentId(material_.num_elements())))
    {
        AtomicNumber z = material_.element_record(elcomp_id).atomic_number();
        WentzelHelper helper(particle_, material_, z, wentzel_, ids_, cutoff_);

        real_type cos_thetamax = helper.cos_thetamax_nuclear();
        if (cos_thetamax < cos_theta)
        {
            result += material_.elements()[elcomp_id.get()].fraction
                      * (helper.calc_xs_nuclear(cos_theta, cos_thetamax)
                         + helper.calc_xs_electron(cos_theta, cos_thetamax));
        }
    }
    result *= material_.number_density();

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
