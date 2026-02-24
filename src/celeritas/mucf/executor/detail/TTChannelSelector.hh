//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/TTChannelSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "celeritas/mucf/interactor/TTMucfInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select final channel for muonic tt molecules.
 *
 * The selection is based on a constant sticking fraction from
 * \citet{bogdanova-mucf-2009, https://doi.org/10.1134/S1063776109020034} ,
 * in which ~14% of the time the muonic alpha channel is selected.
 *
 * \todo Update I/O with user-defined sticking fractions.
 */
class TTChannelSelector
{
  public:
    //!@{
    //! \name Type aliases
    using Channel = TTMucfInteractor::Channel;
    //!@}

    //! Default constructor
    inline CELER_FUNCTION TTChannelSelector() = default;

    // Select fusion channel to be used by the interactor
    template<class Engine>
    inline CELER_FUNCTION Channel operator()(Engine& rng);

  private:
    // Constant sticking fraction of tt fusion
    inline CELER_FUNCTION real_type static constexpr sticking_fraction()
    {
        return 0.14;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Return a selected fusion channel for the \f$ (tt)_\mu \f$ muonic molecule.
 *
 * \sa celeritas::TTMucfInteractor
 */
template<class Engine>
CELER_FUNCTION TTChannelSelector::Channel
TTChannelSelector::operator()(Engine& rng)
{
    Channel result{Channel::size_};

    result = (BernoulliDistribution(this->sticking_fraction())(rng))
                 ? Channel::muonicalpha_neutron_neutron
                 : Channel::alpha_muon_neutron_neutron;

    CELER_ENSURE(result < Channel::size_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
