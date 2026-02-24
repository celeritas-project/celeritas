//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/DTChannelSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "celeritas/mucf/interactor/DTMucfInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select final channel for muonic dt molecules.
 *
 * The selection is based on a constant sticking fraction from
 * \citet{kamimura-mucf-2023, https://doi.org/10.1103/PhysRevC.107.034607}
 * in which ~0.8% of the time the muonic alpha channel is selected.
 *
 * \todo Update I/O with user-defined sticking fractions.
 */
class DTChannelSelector
{
  public:
    //!@{
    //! \name Type aliases
    using Channel = DTMucfInteractor::Channel;
    //!@}

    //! Default constructor
    inline CELER_FUNCTION DTChannelSelector() = default;

    // Select fusion channel to be used by the interactor
    template<class Engine>
    inline CELER_FUNCTION Channel operator()(Engine& rng);

  private:
    // Constant sticking fraction of dt fusion
    inline CELER_FUNCTION static constexpr real_type sticking_fraction()
    {
        return 0.00857;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Return a selected fusion channel for the \f$ (dt)_\mu \f$ muonic molecule.
 *
 * \sa celeritas::DTMucfInteractor
 */
template<class Engine>
CELER_FUNCTION DTChannelSelector::Channel
DTChannelSelector::operator()(Engine& rng)
{
    Channel result{Channel::size_};

    result = (BernoulliDistribution(this->sticking_fraction())(rng))
                 ? Channel::muonicalpha_neutron
                 : Channel::alpha_muon_neutron;

    CELER_ENSURE(result < Channel::size_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
