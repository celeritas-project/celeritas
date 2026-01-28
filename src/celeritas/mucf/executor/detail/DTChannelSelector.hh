//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/DTChannelSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/distribution/GenerateCanonical.hh"
#include "celeritas/mucf/interactor/DTMucfInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select final channel for muonic dt molecules.
 *
 * The selection is a simple selection based on a constant sticking fraction
 * from [ \todo https://arxiv.org/abs/2112.08399 ], in which ~0.8% of the time
 * the muonic alpha channel is selected.
 */
class DTChannelSelector
{
  public:
    //!@{
    //! \name Type aliases
    using Channel = DTMucfInteractor::Channel;
    //!@}

  public:
    //! Default constructor
    inline CELER_FUNCTION DTChannelSelector() = default;

    // Select fusion channel to be used by the interactor
    template<class Engine>
    inline CELER_FUNCTION Channel operator()(Engine& rng);

  private:
    // Constant sticking fraction of dt fusion
    inline CELER_FUNCTION real_type sticking_fraction() const
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

    generate_canonical(rng) > this->sticking_fraction()
        ? result = Channel::alpha_muon_neutron
        : result = Channel::muonicalpha_neutron;

    CELER_ENSURE(result < Channel::size_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
