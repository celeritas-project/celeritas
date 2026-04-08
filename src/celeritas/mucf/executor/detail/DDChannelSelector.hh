//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/DDChannelSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "celeritas/mucf/interactor/DDMucfInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select final channel for muonic dd molecules.
 *
 * The material temperature determines the probability
 * of the outcome of the fusion ending in the \f$ ^3\text{He} \f$ channels
 * versus the tritium channel. If the outcome is a \f$ ^3\text{He} \f$ channel,
 * a constant sticking fraction of 12.2% is used to define if sticking occurs
 * \citet{balin-mucf-2011, https://doi.org/10.1134/S106377961102002X} .
 *
 * \todo Update I/O with user-defined sticking fractions and branching ratios.
 */
class DDChannelSelector
{
  public:
    //!@{
    //! \name Type aliases
    using Channel = DDMucfInteractor::Channel;
    //!@}

    //! Construct with material temperature
    inline CELER_FUNCTION DDChannelSelector(real_type material_temperature);

    // Select fusion channel to be used by the interactor
    template<class Engine>
    inline CELER_FUNCTION Channel operator()(Engine& rng);

  private:
    real_type he3_probability_;

    // Constant sticking fraction between the two 3He channels
    inline CELER_FUNCTION static constexpr real_type sticking_fraction()
    {
        return 0.122;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with material temperature.
 *
 * The temperature is used to calculate the branching ratio between either of
 * the \f$ ^3\text{He} \f$ channels and the tritium channel.
 */
CELER_FUNCTION
DDChannelSelector::DDChannelSelector(real_type material_temperature)
{
    CELER_EXPECT(material_temperature > 0);

    real_type branching_ratio{0};
    if (material_temperature <= 50)
    {
        branching_ratio = 1;
    }
    else if (material_temperature <= 100)
    {
        branching_ratio = 1.0088 * (material_temperature - 50);
    }
    else
    {
        branching_ratio = 1.44;
    }
    CELER_ASSERT(branching_ratio > 0);

    he3_probability_ = branching_ratio / (branching_ratio + 1);
    CELER_ENSURE(he3_probability_ > 0 && he3_probability_ <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Return a selected fusion channel for the \f$ (dd)_\mu \f$ muonic molecule.
 *
 * \sa celeritas::DDMucfInteractor
 */
template<class Engine>
CELER_FUNCTION DDChannelSelector::Channel
DDChannelSelector::operator()(Engine& rng)
{
    Channel result{Channel::size_};

    result = BernoulliDistribution(he3_probability_)(rng)
                 // Select between the two 3He channels
                 ? (BernoulliDistribution(this->sticking_fraction())(rng)
                        ? Channel::muonichelium3_neutron
                        : Channel::helium3_muon_neutron)
                 // Select tritium channel
                 : Channel::tritium_muon_proton;

    CELER_ENSURE(result < Channel::size_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
