//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/DeltaDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Distribution where the "sampled" value is just the given value.
 */
template<class T>
class DeltaDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using value_type = T;
    //!@}

  public:
    // Constructor
    explicit CELER_FUNCTION DeltaDistribution(value_type value) : value_(value)
    {
    }

    // "Sample" the value
    template<class Generator>
    CELER_FUNCTION value_type operator()(Generator&) const
    {
        return value_;
    }

  private:
    value_type value_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
