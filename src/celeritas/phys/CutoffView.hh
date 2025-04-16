//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/CutoffView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "CutoffData.hh"
#include "Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access invariant material- and particle-dependent cutoff values.
 *
 * \c CutoffParamsData is defined in \c CutoffData.hh and constructed by
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
    //! \name Type aliases
    using CutoffId = OpaqueId<ParticleCutoff>;
    using CutoffData = NativeCRef<CutoffParamsData>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct for the given particle and material ids
    inline CELER_FUNCTION
    CutoffView(CutoffData const& params, PhysMatId material);

    // Return energy cutoff value
    inline CELER_FUNCTION Energy energy(ParticleId particle) const;

    // Return range cutoff value
    inline CELER_FUNCTION real_type range(ParticleId particle) const;

    // Whether to kill secondaries below the production cut post-interaction
    inline CELER_FUNCTION bool apply_post_interaction() const;

    // Whether the post-interaction cutoff should be applied to the secondary
    inline CELER_FUNCTION bool apply(Secondary const&) const;

  private:
    CutoffData const& params_;
    PhysMatId material_;

    //// HELPER FUNCTIONS ////

    // Get the cutoff for the given particle and material
    CELER_FORCEINLINE_FUNCTION ParticleCutoff get(ParticleId particle) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct view from host/device for the given material id.
 */
CELER_FUNCTION
CutoffView::CutoffView(CutoffData const& params, PhysMatId material)
    : params_(params), material_(material)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(material_ < params_.num_materials);
}

//---------------------------------------------------------------------------//
/*!
 * Return energy cutoff value.
 */
CELER_FUNCTION auto CutoffView::energy(ParticleId particle) const -> Energy
{
    return this->get(particle).energy;
}

//---------------------------------------------------------------------------//
/*!
 * Return range cutoff value.
 */
CELER_FUNCTION real_type CutoffView::range(ParticleId particle) const
{
    return this->get(particle).range;
}

//---------------------------------------------------------------------------//
/*!
 * Whether to kill secondaries below the production cut post-interaction.
 */
CELER_FUNCTION bool CutoffView::apply_post_interaction() const
{
    return params_.apply_post_interaction;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the post-interaction cutoff should be applied to the secondary.
 *
 * This will be true if the \c apply_post_interaction option is enabled and the
 * seccondary is an electron, positron, or gamma with energy below the
 * production cut.
 */
CELER_FUNCTION bool CutoffView::apply(Secondary const& secondary) const
{
    return (secondary.particle_id == params_.ids.gamma
            || secondary.particle_id == params_.ids.electron
            || secondary.particle_id == params_.ids.positron)
           && secondary.energy < this->energy(secondary.particle_id);
}

//---------------------------------------------------------------------------//
/*!
 * Get the cutoff for the given particle and material.
 */
CELER_FUNCTION ParticleCutoff CutoffView::get(ParticleId particle) const
{
    CELER_EXPECT(particle < params_.id_to_index.size());
    CELER_EXPECT(params_.id_to_index[particle] < params_.num_particles);
    CutoffId id{params_.num_materials * params_.id_to_index[particle]
                + material_.get()};
    CELER_ENSURE(id < params_.cutoffs.size());
    return params_.cutoffs[id];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
