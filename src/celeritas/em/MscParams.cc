//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MscParams.cc
//---------------------------------------------------------------------------//
#include "MscParams.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridInserter.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/io/ImportModel.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Default destructor
MscParams::~MscParams() = default;

//---------------------------------------------------------------------------//
/*!
 * Validate and save MSC IDs.
 */
void MscParams::build_ids(MscIds* ids, ParticleParams const& particles) const
{
    ids->electron = particles.find(pdg::electron());
    ids->positron = particles.find(pdg::positron());
    CELER_VALIDATE(ids->electron && ids->positron,
                   << "missing e-/e+ (required for MSC)");

    // TODO: change IDs to a vector for all particles. This should apply
    // to muons and charged hadrons as well
    if (particles.find(pdg::mu_minus()) || particles.find(pdg::mu_plus())
        || particles.find(pdg::proton()))
    {
        CELER_LOG(warning) << "Multiple scattering is not implemented for for "
                              "particles other than electron and positron";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build the scaled cross section data.
 */
void MscParams::build_xs(XsValues* scaled_xs,
                         Values* reals,
                         ParticleParams const& particles,
                         MaterialParams const& materials,
                         VecImportMscModel const& mdata_vec,
                         ImportModelClass model_class) const
{
    // Filter MSC data by model and particle type
    std::vector<ImportMscModel const*> msc_data(particles.size(), nullptr);
    for (ImportMscModel const& imm : mdata_vec)
    {
        // Filter out other MSC models
        if (imm.model_class != model_class)
            continue;

        // Filter out unused particles
        PDGNumber pdg{imm.particle_pdg};
        ParticleId pid = pdg ? particles.find(pdg) : ParticleId{};
        if (!pid)
            continue;

        if (!msc_data[pid.get()])
        {
            // Save data
            msc_data[pid.get()] = &imm;
        }
        else
        {
            // Warn: possibly multiple physics lists or different models in
            // different regions
            CELER_LOG(warning)
                << "duplicate " << to_cstring(imm.model_class)
                << " physics data for particle " << particles.id_to_label(pid)
                << ": ignoring all but the first encountered model";
        }
    }

    auto get_scaled_xs = [&msc_data, &particles](ParticleId pid) {
        CELER_ASSERT(pid < msc_data.size());
        ImportMscModel const* imm = msc_data[pid.unchecked_get()];
        CELER_VALIDATE(imm,
                       << "missing " << to_cstring(imm->model_class)
                       << " physics data for particle "
                       << particles.id_to_label(pid));
        CELER_ASSERT(imm->xs_table.x_units == ImportUnits::mev);
        CELER_ASSERT(imm->xs_table.y_units == ImportUnits::mev_2_per_cm);
        return &imm->xs_table;
    };

    // Particle-dependent data
    Array<ParticleId, 2> const par_ids{
        {particles.find(pdg::electron()), particles.find(pdg::positron())}};
    Array<ImportPhysicsTable const*, 2> const xs_tables{{
        get_scaled_xs(par_ids[0]),
        get_scaled_xs(par_ids[1]),
    }};

    // Get initial high/low energy limits to validate energy grids
    auto const& phys_vec = get_scaled_xs(par_ids[0])->physics_vectors;
    CELER_ASSERT(!phys_vec.empty() && phys_vec[0]);
    real_type energy_min = phys_vec[0].x.front();
    real_type energy_max = phys_vec[0].x.back();

    // Scaled cross section builder
    CollectionBuilder xs(scaled_xs);
    xs.reserve(2 * materials.num_materials());

    // TODO: simplify when refactoring ValueGridInserter, etc
    ValueGridInserter::XsGridCollection xgc;
    ValueGridInserter vgi{reals, &xgc};

    for (size_type mat_idx : range(materials.num_materials()))
    {
        for (size_type par_idx : range(par_ids.size()))
        {
            // Get the cross section data for this particle and material
            ImportPhysicsVector const& pvec
                = xs_tables[par_idx]->physics_vectors[mat_idx];
            CELER_ASSERT(pvec.vector_type == ImportPhysicsVectorType::log);

            // Check that the limits are the same for all materials and
            // particles; otherwise we need to change \c *Msc::is_applicable to
            // look up the particle and material
            CELER_VALIDATE(energy_min == real_type(pvec.x.front())
                               && energy_max == real_type(pvec.x.back()),
                           << "multiple scattering cross section energy "
                              "limits are inconsistent across particles "
                              "and/or materials");

            // To reuse existing code (TODO: simplify when refactoring)
            // use the value grid builder to construct the grid entry in a
            // temporary container and then copy it into the pm data.
            auto vgb = ValueGridLogBuilder::from_geant(make_span(pvec.x),
                                                       make_span(pvec.y));
            auto grid_id = vgb->build(vgi);
            CELER_ASSERT(grid_id.get() == xs.size());
            xs.push_back(xgc[grid_id]);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
