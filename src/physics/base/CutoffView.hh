//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "CutoffData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access invariant material- and particle-dependent cutoff values.
 *
 * \c CutoffParamsData is defined in \c CutoffInterface and constructed by
 * \c CutoffParams .
 *
 * \code
 * CutoffParams cutoffs(input);
 * CutoffView cutoff_view(cutoffs.host_ref(), material_id);
 * cutoff_view.energy(particle_id);
 * cutoff_view.range(particle_id);
 * \endcode
 */
class CutoffView
{
  public:
    //!@{
    //! Type aliases
    using CutoffId = OpaqueId<ParticleCutoff>;
    using CutoffData
        = CutoffParamsData<Ownership::const_reference, MemSpace::native>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct for the given particle and material ids
    inline CELER_FUNCTION
    CutoffView(const CutoffData& params, MaterialId material);

    // Return energy cutoff value
    inline CELER_FUNCTION Energy energy(ParticleId particle) const;

    // Return range cutoff value
    inline CELER_FUNCTION real_type range(ParticleId particle) const;

  private:
    const CutoffData& params_;
    MaterialId        material_;

    //// HELPER FUNCTIONS ////

    // Get the cutoff for the given particle and material
    CELER_FORCEINLINE_FUNCTION ParticleCutoff get(ParticleId particle) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "CutoffView.i.hh"
