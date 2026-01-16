//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/DDChannelSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/mucf/interactor/DDMucfInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select final channel for muonic dd molecules.
 *
 * This selection already accounts for sticking, as that is one of the possible
 * outcomes.
 */
class DDChannelSelector
{
  public:
    //!@{
    //! \name Type aliases
    using Channel = DDMucfInteractor::Channel;
    //!@}

  public:
    //! Construct with args; \todo Update documentation
    inline CELER_FUNCTION DDChannelSelector(/* args */);

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
CELER_FUNCTION DDChannelSelector::DDChannelSelector(/* args */)
{
    CELER_NOT_IMPLEMENTED("Mucf dd fusion channel selection");
}

//---------------------------------------------------------------------------//
/*!
 * Return a sampled channel to be used as input in the dd muCF interactor.
 *
 * \sa celeritas::DDMucfInteractor
 */
template<class Engine>
CELER_FUNCTION DDChannelSelector::Channel DDChannelSelector::operator()(Engine&)
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
