//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/detail/MscParamsHelper.cc
//---------------------------------------------------------------------------//
#include "MscParamsHelper.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridInserter.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/io/ImportModel.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data and particle and material properties.
 */
MscParamsHelper::MscParamsHelper(ParticleParams const& particles,
                                 VecImportMscModel const& mdata,
                                 ImportModelClass model_class)
    : particles_(particles)
    , model_class_(model_class)
    , pid_to_xs_(particles_.size())
{
    // Filter MSC data by model and particle type
    for (ImportMscModel const& imm : mdata)
    {
        // Filter out other MSC models
        if (imm.model_class != model_class_)
            continue;

        // Filter out unused particles
        PDGNumber pdg{imm.particle_pdg};
        ParticleId pid = pdg ? particles_.find(pdg) : ParticleId{};
        if (!pid)
            continue;

        if (!pid_to_xs_[pid.get()])
        {
            // Save mapping of particle ID to index in cross section table
            pid_to_xs_[pid.get()] = MscParticleId(xs_tables_.size());
            CELER_LOG(debug) << "found " << to_cstring(imm.model_class)
                             << " physics data for particle "
                             << particles_.id_to_label(pid);
        }
        else
        {
            // Warn: possibly multiple physics lists or different models in
            // different regions
            CELER_LOG(warning)
                << "duplicate " << to_cstring(imm.model_class)
                << " physics data for particle " << particles_.id_to_label(pid)
                << ": ignoring all but the first encountered model";
        }

        // Save particle ID and scaled cross section table
        par_ids_.push_back(pid);
        CELER_ASSERT(imm.xs_table.x_units == ImportUnits::mev);
        CELER_ASSERT(imm.xs_table.y_units == ImportUnits::mev_2_per_cm);
        xs_tables_.push_back(&imm.xs_table);
    }
    CELER_VALIDATE(!xs_tables_.empty(),
                   << "missing physics data for " << to_cstring(model_class));
}

//---------------------------------------------------------------------------//
/*!
 * Validate and save MSC IDs.
 */
void MscParamsHelper::build_ids(CoulombIds* ids, IndexValues* pid_to_xs) const
{
    ids->electron = particles_.find(pdg::electron());
    ids->positron = particles_.find(pdg::positron());
    CELER_VALIDATE(ids->electron && ids->positron,
                   << "missing e-/e+ (required for MSC)");

    make_builder(pid_to_xs).insert_back(pid_to_xs_.begin(), pid_to_xs_.end());
}

//---------------------------------------------------------------------------//
/*!
 * Build the macroscopic cross section scaled by energy squared.
 */
void MscParamsHelper::build_xs(XsValues* scaled_xs, Values* reals) const
{
    // Scaled cross section builder
    CollectionBuilder xs(scaled_xs);
    size_type num_materials = xs_tables_[0]->physics_vectors.size();
    xs.reserve(par_ids_.size() * num_materials);

    // TODO: simplify when refactoring ValueGridInserter, etc
    ValueGridInserter::XsGridCollection xgc;
    ValueGridInserter vgi{reals, &xgc};

    for (size_type mat_idx : range(num_materials))
    {
        for (size_type par_idx : range(par_ids_.size()))
        {
            // Get the cross section data for this particle and material
            CELER_ASSERT(pid_to_xs_[par_ids_[par_idx].get()].get() == par_idx);
            CELER_ASSERT(mat_idx < xs_tables_[par_idx]->physics_vectors.size());
            ImportPhysicsVector const& pvec
                = xs_tables_[par_idx]->physics_vectors[mat_idx];
            CELER_ASSERT(pvec.vector_type == ImportPhysicsVectorType::log);

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
/*!
 * Get the cross section table energy grid bounds.
 *
 * This expects the grid bounds to be the same for all particles and materials.
 */
auto MscParamsHelper::energy_grid_bounds() const -> EnergyBounds
{
    EnergyBounds result;
    {
        // Get initial high/low energy limits
        CELER_ASSERT(!xs_tables_[0]->physics_vectors.empty());
        auto const& pvec = xs_tables_[0]->physics_vectors[0];
        CELER_ASSERT(pvec);
        result = {Energy(pvec.x.front()), Energy(pvec.x.back())};
    }
    for (size_type par_idx : range(par_ids_.size()))
    {
        auto const& phys_vectors = xs_tables_[par_idx]->physics_vectors;
        for (auto const& pvec : phys_vectors)
        {
            // Check that the limits are the same for all materials and
            // particles; otherwise we need to change \c *Msc::is_applicable to
            // look up the particle and material
            CELER_VALIDATE(result[0].value() == real_type(pvec.x.front())
                               && result[1].value() == real_type(pvec.x.back()),
                           << "multiple scattering cross section energy "
                              "limits are inconsistent across particles "
                              "and/or materials");
        }
    }
    CELER_ENSURE(result[0] < result[1]);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
