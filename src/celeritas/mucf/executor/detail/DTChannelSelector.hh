//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/DTChannelSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/mucf/interactor/DTMucfInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select final channel for muonic dt molecules.
 *
 * This selection already accounts for sticking, as that is one of the possible
 * outcomes.
 */
class DTChannelSelector
{
  public:
    //!@{
    //! \name Type aliases
    using Channel = DTMucfInteractor::Channel;
    //!@}

  public:
    //! Construct with args; \todo Update documentation
    inline CELER_FUNCTION DTChannelSelector(/* args */);

    // Select fusion channel to be used by the interactor
    template<class Engine>
    inline CELER_FUNCTION Channel operator()(Engine&);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with args.
 *
 * \todo Update documentation
 */
CELER_FUNCTION DTChannelSelector::DTChannelSelector(/* args */)
{
    CELER_NOT_IMPLEMENTED("Mucf dt fusion channel selection");
}

//---------------------------------------------------------------------------//
/*!
 * Return a sampled channel to be used as input in the dt muCF interactor.
 *
 * \sa celeritas::DTMucfInteractor
 */
template<class Engine>
CELER_FUNCTION DTChannelSelector::Channel DTChannelSelector::operator()(Engine&)
{
    Channel result{Channel::size_};

    //! \todo Implement
    // Final channel selection already takes into account sticking.

    CELER_ENSURE(result < Channel::size_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
