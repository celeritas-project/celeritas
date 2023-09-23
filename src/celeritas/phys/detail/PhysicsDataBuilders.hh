//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/PhysicsDataBuilders.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionBuilder.hh"

#include "../PhysicsData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construction helpers for PhysicsParamsData.
 */
struct PhysicsDataBuilders
{
    //// TYPES ////

    template<class T>
    using Builder = CollectionBuilder<T, MemSpace::host>;
    template<class T>
    using ParticleBuilder = CollectionBuilder<T, MemSpace::host, ParticleId>;
    template<class T>
    using ParticleModelBuilder
        = CollectionBuilder<T, MemSpace::host, ParticleModelId>;

    //// BUILDERS ////

    HardwiredModels<Ownership::value, MemSpace::host>* hardwired;
    PhysicsParamsScalars* scalars;

    Builder<real_type> reals;
    Builder<ParticleModelId> pmodel_ids;
    Builder<ValueGrid> value_grids;
    Builder<ValueGridId> value_grid_ids;
    Builder<ProcessId> process_ids;
    Builder<ValueTable> value_tables;
    Builder<ValueTableId> value_table_ids;
    Builder<IntegralXsProcess> integral_xs;
    Builder<ModelGroup> model_groups;
    ParticleBuilder<ProcessGroup> process_groups;
    ParticleModelBuilder<ModelId> model_ids;
    ParticleModelBuilder<ModelXsTable> model_xs;

    ValueGridInserter insert_grid;

    //// CONSTRUCTOR ////

    explicit PhysicsDataBuilders(
        PhysicsParamsData<Ownership::value, MemSpace::host>* data)
        : hardwired(&data->hardwired)
        , scalars(&data->scalars)
        , reals(&data->reals)
        , pmodel_ids(&data->pmodel_ids)
        , value_grids(&data->value_grids)
        , value_grid_ids(&data->value_grid_ids)
        , process_ids(&data->process_ids)
        , value_tables(&data->value_tables)
        , value_table_ids(&data->value_table_ids)
        , integral_xs(&data->integral_xs)
        , model_groups(&data->model_groups)
        , process_groups(&data->process_groups)
        , model_ids(&data->model_ids)
        , model_xs(&data->model_xs)
        , insert_grid(&data->reals, &data->value_grids)
    {
        CELER_EXPECT(data);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
