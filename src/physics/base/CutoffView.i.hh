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
 * Construct view from host/device for the given particle and material ids.
 */
CutoffView::CutoffView(const CutoffPointers& params,
                       ParticleId            particle,
                       MaterialId            material)
{
    CELER_EXPECT(particle < params.num_particles);
    CELER_EXPECT(material < params.num_materials);
    using CutoffId = OpaqueId<ParticleCutoff>;
    CutoffId cutoff_id{params.num_materials * particle.get() + material.get()};

    cutoff_ = params.cutoffs[cutoff_id];
}
//---------------------------------------------------------------------------//
} // namespace celeritas
