//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffView.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct view from host/device for the given material id.
 */
CELER_FUNCTION
CutoffView::CutoffView(const CutoffData& params, MaterialId material)
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
    return params_.cutoffs[this->index(particle)].energy;
}

//---------------------------------------------------------------------------//
/*!
 * Return range cutoff value.
 */
CELER_FUNCTION real_type CutoffView::range(ParticleId particle) const
{
    return params_.cutoffs[this->index(particle)].range;
}

//---------------------------------------------------------------------------//
/*!
 * Index in cutoffs of the given particle and material.
 */
CELER_FUNCTION auto CutoffView::index(ParticleId particle) const -> CutoffId
{
    CELER_EXPECT(particle < params_.id_to_index.size());
    CELER_EXPECT(params_.id_to_index[particle] < params_.num_particles);
    CutoffId id{params_.num_materials * params_.id_to_index[particle]
                + material_.get()};
    CELER_ENSURE(id < params_.cutoffs.size());
    return id;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
