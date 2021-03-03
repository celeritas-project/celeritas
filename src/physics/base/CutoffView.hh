//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "CutoffInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access invariant material and particle dependent cutoff values.
 */
class CutoffView
{
  public:
    //!@{
    //! Type aliases
    using CutoffPointers
        = CutoffParamsData<Ownership::const_reference, MemSpace::native>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from
    inline CELER_FUNCTION CutoffView(const CutoffPointers& params,
                                     ParticleId            particle,
                                     MaterialId            material);

    CELER_FORCEINLINE_FUNCTION Energy energy() const
    {
        return cutoffs_.energy;
    }

    CELER_FORCEINLINE_FUNCTION real_type range() const
    {
        return cutoffs_.range;
    }

  private:
    SingleCutoff cutoffs_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "CutoffView.i.hh"
