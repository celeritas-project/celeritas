//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Collection.hh"
#include "Types.hh"
#include "physics/grid/XsGridInterface.hh"
#include "physics/em/detail/LivermorePE.hh"
#include "physics/em/detail/EPlusGG.hh"
#include "physics/material/Types.hh"

#ifndef __CUDA_ARCH__
#    include "base/CollectionBuilder.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
//! Currently all value grids are cross section grids
using ValueGrid    = XsGridData;
using ValueGridId  = OpaqueId<XsGridData>;
using ValueTableId = OpaqueId<struct ValueTable>;

//---------------------------------------------------------------------------//
//! Hardcoded types of grid data
enum class ValueGridType
{
    macro_xs,    //!< Interaction cross sections
    energy_loss, //!< Energy loss per unit length
    range,       //!< Particle range
    size_        //!< Sentinel value
};

template<class T>
using ValueGridArray = Array<T, size_type(ValueGridType::size_)>;

//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Energy-dependent model IDs for a single process and particle type.
 *
 * For a given particle type, a single process should be divided into multiple
 * models as a function of energy. The ModelGroup represents this with an
 * energy grid, and each cell of the grid corresponding to a particular
 * ModelId.
 */
struct ModelGroup
{
    ItemRange<real_type> energy; //!< Energy grid bounds [MeV]
    ItemRange<ModelId>   model;  //!< Corresponding models

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return (energy.size() >= 2) && (model.size() + 1 == energy.size());
    }
};

//---------------------------------------------------------------------------//
/*!
 * Set of value grids for all materials.
 *
 * It is allowable for this to be "false" (i.e. no materials assigned)
 * indicating that the value table doesn't apply in the context -- for
 * example, an empty ValueTable macro_xs means that the process doesn't have a
 * discrete interaction.
 */
struct ValueTable
{
    ItemRange<ValueGridId> material; //!< Value grid by material index

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !material.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Processes for a single particle type.
 *
 * Each index should be accessed with type ParticleProcessId. The "tables" are
 * a fixed-size number of ItemRange references to ValueTables. The first index
 * of the table (hard-coded) corresponds to ValueGridType; the second index is
 * a ParticleProcessId. So the cross sections for ParticleProcessId{2} would
 * be \code tables[size_type(ValueGridType::macro_xs)][2] \endcode. This
 * awkward access is encapsulated by the PhysicsTrackView.
 */
struct ProcessGroup
{
    ItemRange<ProcessId> processes; //!< Processes that apply [ppid]
    ValueGridArray<ItemRange<ValueTable>> tables; //!< [vgt][ppid]
    ItemRange<ModelGroup> models; //!< Model applicability [ppid]

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !processes.empty() && models.size() == processes.size();
    }

    //! Number of processes that apply
    CELER_FUNCTION ParticleProcessId::size_type size() const
    {
        return processes.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Model data for special hardwired cases (on-the-fly xs calculations).
 */
struct HardwiredModels
{
    // Photoelectric effect
    ProcessId                   photoelectric;
    units::MevEnergy            photoelectric_table_thresh;
    ModelId                     livermore_pe;
    detail::LivermorePEPointers livermore_pe_params;

    // Positron annihilation
    ProcessId               positron_annihilation;
    ModelId                 eplusgg;
    detail::EPlusGGPointers eplusgg_params;
};

//---------------------------------------------------------------------------//
/*!
 * Persistent shared physics data.
 *
 * This includes macroscopic cross section, energy loss, and range tables
 * ordered by [particle][process][material][energy].
 *
 * So the first applicable process (ProcessId{0}) for an arbitrary particle
 * (ParticleId{1}) in material 2 (MaterialId{2}) will have the following
 * ID and cross section grid: \code
   ProcessId proc_id = params.particle[1].processes[0];
   const UniformGridData& grid
       =
 params.particle[1].table[int(ValueGridType::macro_xs)][0].material[2].log_energy;
 * \endcode
 */
template<Ownership W, MemSpace M>
struct PhysicsParamsData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ParticleItems = Collection<T, W, M, ParticleId>;

    // Backend storage
    Items<real_type>            reals;
    Items<ModelId>              model_ids;
    Items<ValueGrid>            value_grids;
    Items<ValueGridId>          value_grid_ids;
    Items<ProcessId>            process_ids;
    Items<ValueTable>           value_tables;
    Items<ModelGroup>           model_groups;
    ParticleItems<ProcessGroup> process_groups;

    HardwiredModels       hardwired;
    ProcessId::size_type  max_particle_processes{};

    //// USER-CONFIGURABLE CONSTANTS ////
    real_type scaling_min_range{}; //!< rho [cm]
    real_type scaling_fraction{};  //!< alpha [unitless]
    real_type linear_loss_limit{}; //!< For scaled range calculation

    //// MEMBER FUNCTIONS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !process_groups.empty() && max_particle_processes
               && scaling_min_range > 0 && scaling_fraction > 0
               && linear_loss_limit > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    PhysicsParamsData& operator=(const PhysicsParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        reals          = other.reals;
        model_ids      = other.model_ids;
        value_grids    = other.value_grids;
        value_grid_ids = other.value_grid_ids;
        process_ids    = other.process_ids;
        value_tables   = other.value_tables;
        model_groups   = other.model_groups;
        process_groups = other.process_groups;

        hardwired              = other.hardwired;
        max_particle_processes = other.max_particle_processes;

        scaling_min_range = other.scaling_min_range;
        scaling_fraction  = other.scaling_fraction;
        linear_loss_limit = other.linear_loss_limit;

        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Physics state data for a single track.
 *
 * - Remaining number of mean free paths to the next discrete interaction
 * - Maximum step length (limited by range, energy loss, and interaction)
 * - Selected model ID if undergoing an interaction
 */
struct PhysicsTrackState
{
    real_type interaction_mfp; //!< Remaining MFP to interaction
    real_type step_length;     //!< Overall physics step length
    real_type macro_xs;        //!< Total cross section

    ModelId            model_id;   //!< Selected model if interacting
    ElementComponentId element_id; //!< Selected element during interaction
};

//---------------------------------------------------------------------------//
/*!
 * Initialize a physics track state.
 *
 * Currently no data is required at initialization -- it all must be evaluated
 * by the physics kernels itself.
 */
struct PhysicsTrackInitializer
{
};

//---------------------------------------------------------------------------//
/*!
 * Dynamic physics (models, processes) state data.
 *
 * The "xs scratch space" is a 2D array of reals, indexed with
 * [track_id][el_component_id], where the fast-moving dimension has the
 * greatest number of element components of any material in the problem. This
 * can be used for the physics to calculate microscopic cross sections.
 */
template<Ownership W, MemSpace M>
struct PhysicsStateData
{
    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    template<class T>
    using Items = celeritas::Collection<T, W, M>;

    StateItems<PhysicsTrackState> state; //!< Track state [track]
    Items<real_type> per_process_xs;     //!< XS [track][particle process]

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    PhysicsStateData& operator=(PhysicsStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        state          = other.state;
        per_process_xs = other.per_process_xs;
        return *this;
    }
};

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
/*!
 * Resize a material state in host code.
 */
template<MemSpace M>
inline void resize(
    PhysicsStateData<Ownership::value, M>*                               state,
    const PhysicsParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type                                                            size)
{
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params.max_particle_processes > 0);
    make_builder(&state->state).resize(size);
    make_builder(&state->per_process_xs)
        .resize(size * params.max_particle_processes);
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
