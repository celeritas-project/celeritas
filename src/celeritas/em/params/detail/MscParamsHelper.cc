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
#include "corecel/grid/VectorUtils.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/grid/UniformGridInserter.hh"
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
    size_type num_materials = xs_tables_[0]->grids.size();
    xs.reserve(par_ids_.size() * num_materials);

    // TODO: simplify when refactoring GridInserter, etc
    UniformGridInserter::GridValues grids;
    UniformGridInserter insert{reals, &grids};

    for (size_type mat_idx : range(num_materials))
    {
        for (size_type par_idx : range(par_ids_.size()))
        {
            CELER_ASSERT(pid_to_xs_[par_ids_[par_idx].get()].get() == par_idx);
            CELER_ASSERT(mat_idx < xs_tables_[par_idx]->grids.size());

            // Get the cross section data for this particle and material
            auto const& grid = xs_tables_[par_idx]->grids[mat_idx];
            CELER_ASSERT(grid && std::exp(grid.x[Bound::lo]) > 0);

            auto grid_id = insert(grid);
            CELER_ASSERT(grid_id.get() == xs.size());

            xs.push_back(grids[grid_id]);
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
    auto x = [this] {
        // Get initial high/low energy limits
        CELER_ASSERT(!xs_tables_[0]->grids.empty());
        auto const& grid = xs_tables_[0]->grids[0];
        CELER_ASSERT(grid);
        return grid.x;
    }();
    for (size_type par_idx : range(par_ids_.size()))
    {
        auto const& phys_vectors = xs_tables_[par_idx]->grids;
        for (auto const& grid : phys_vectors)
        {
            // Check that the limits are the same for all materials and
            // particles; otherwise we need to change \c *Msc::is_applicable to
            // look up the particle and material
            CELER_VALIDATE(x[Bound::lo] == grid.x[Bound::lo]
                               && x[Bound::hi] == grid.x[Bound::hi],
                           << "multiple scattering cross section energy "
                              "limits are inconsistent across particles "
                              "and/or materials");
        }
    }
    return {Energy(std::exp(x[Bound::lo])), Energy(std::exp(x[Bound::hi]))};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
