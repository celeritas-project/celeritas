//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/executor/detail/TTChannelSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/mucf/interactor/TTMucfInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select final channel for muonic tt molecules.
 *
 * This selection already accounts for sticking, as that is one of the possible
 * outcomes.
 */
class TTChannelSelector
{
  public:
    //!@{
    //! \name Type aliases
    using Channel = TTMucfInteractor::Channel;
    //!@}

  public:
    //! Construct with args; \todo Update documentation
    inline CELER_FUNCTION TTChannelSelector(/* args */);

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
CELER_FUNCTION TTChannelSelector::TTChannelSelector(/* args */)
{
    //! \todo Implement
    CELER_NOT_IMPLEMENTED("Mucf tt fusion channel selection");
}

//---------------------------------------------------------------------------//
/*!
 * Return a sampled channel to be used as input in the tt muCF interactor.
 *
 * \sa celeritas::TTMucfInteractor
 */
template<class Engine>
CELER_FUNCTION TTChannelSelector::Channel TTChannelSelector::operator()(Engine&)
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
