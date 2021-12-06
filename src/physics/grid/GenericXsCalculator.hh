//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GenericXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate cross sections on a nonuniform grid.
 */
class GenericXsCalculator
{
  public:
    //@{
    //! Type aliases
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //@}

  public:
    // Construct from grid data and backend values
    inline CELER_FUNCTION
    GenericXsCalculator(const GenericGridData& grid, const Values& values);

    // Find and interpolate the cross section from the given energy
    inline CELER_FUNCTION real_type operator()(const real_type energy) const;

  private:
    const GenericGridData& data_;
    const Values&          reals_;

    CELER_FORCEINLINE_FUNCTION real_type get(size_type index) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "GenericXsCalculator.i.hh"
