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
CutoffView::CutoffView(const CutoffPointers& params, MaterialId material)
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
    CutoffId id{params_.num_materials * particle.get() + material_.get()};
    CELER_ASSERT(id < params_.cutoffs.size());
    return params_.cutoffs[id].energy;
}

//---------------------------------------------------------------------------//
/*!
 * Return range cutoff value.
 */
CELER_FUNCTION real_type CutoffView::range(ParticleId particle) const
{
    CutoffId id{params_.num_materials * particle.get() + material_.get()};
    CELER_ASSERT(id < params_.cutoffs.size());
    return params_.cutoffs[id].range;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
