//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/grid/ValueGridData.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/Types.hh"

#include "Interaction.hh"
#include "Secondary.hh"

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
    using Energy = units::MevEnergy;

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
 * Energy loss process that uses MC integration to sample interaction length.
 *
 * This is needed for the integral approach for correctly sampling the discrete
 * interaction length after a particle loses energy along a step. An \c
 * IntegralXsProcess is stored for each particle-process. This will be "false"
 * (i.e. no energy_max assigned) if the process is not continuous-discrete or
 * if \c use_integral_xs is false.
 */
struct IntegralXsProcess
{
    ItemRange<real_type> energy_max_xs; //!< Energy of the largest xs [mat]

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !energy_max_xs.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Processes for a single particle type.
 *
 * Each index should be accessed with type ParticleProcessId. The "tables" are
 * a fixed-size number of ItemRange references to ValueTables. The first index
 * of the table (hard-coded) corresponds to ValueGridType; the second index is
 * a ParticleProcessId. So the cross sections for ParticleProcessId{2} would
 * be \code tables[ValueGridType::macro_xs][2] \endcode. This
 * awkward access is encapsulated by the PhysicsTrackView. \c integral_xs will
 * only be assigned if the integral approach is used and the particle has
 * continuous-discrete processes.
 */
struct ProcessGroup
{
    ItemRange<ProcessId> processes; //!< Processes that apply [ppid]
    ValueGridArray<ItemRange<ValueTable>> tables;      //!< [vgt][ppid]
    ItemRange<IntegralXsProcess>          integral_xs; //!< [ppid]
    ItemRange<ModelGroup> models;   //!< Model applicability [ppid]
    ParticleProcessId eloss_ppid{}; //!< Process with de/dx and range tables
    ParticleProcessId msc_ppid{};   //!< Process of msc
    bool has_at_rest{}; //!< Whether the particle type has an at-rest process

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
template<Ownership W, MemSpace M>
struct HardwiredModels
{
    // Photoelectric effect
    ProcessId                     photoelectric;
    units::MevEnergy              photoelectric_table_thresh;
    ModelId                       livermore_pe;
    detail::LivermorePEData<W, M> livermore_pe_data;

    // Positron annihilation
    ProcessId           positron_annihilation;
    ModelId             eplusgg;
    detail::EPlusGGData eplusgg_data;

    // Multiple scattering (data for the mean free path)
    ProcessId                  msc;
    ModelId                    urban;
    detail::UrbanMscData<W, M> urban_data;

    //// MEMBER FUNCTIONS ////

    //! Assign from another set of hardwired models
    template<Ownership W2, MemSpace M2>
    HardwiredModels& operator=(const HardwiredModels<W2, M2>& other)
    {
        // Note: don't require the other set of hardwired models to be assigned
        photoelectric = other.photoelectric;
        if (photoelectric)
        {
            // Only assign photoelectric data if that process is present
            photoelectric_table_thresh = other.photoelectric_table_thresh;
            livermore_pe               = other.livermore_pe;
            livermore_pe_data          = other.livermore_pe_data;
        }
        positron_annihilation = other.positron_annihilation;
        eplusgg               = other.eplusgg;
        eplusgg_data          = other.eplusgg_data;

        msc = other.msc;
        if (msc)
        {
            // Only assign msc data if that process is present
            urban      = other.urban;
            urban_data = other.urban_data;
        }

        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Scalar (no template needed) quantities used by physics.
 *
 * The user-configurable constants are described in \c PhysicsParams .
 *
 * The \c model_to_action value corresponds to the \c ActionId for the first \c
 * ModelId . Additionally it implies (by construction in physics_params) the
 * action IDs of several other physics actions.
 */
struct PhysicsParamsScalars
{
    //! Highest number of processes for any particle type
    ProcessId::size_type max_particle_processes{};
    //! Offset to create an ActionId from a ModelId
    ActionId::size_type model_to_action{};
    //! Number of physics models
    ModelId::size_type num_models{};

    // User-configurable constants
    real_type scaling_min_range{};  //!< rho [cm]
    real_type scaling_fraction{};   //!< alpha [unitless]
    real_type energy_fraction{};    //!< xi [unitless]
    real_type linear_loss_limit{};  //!< For scaled range calculation
    real_type fixed_step_limiter{}; //!< Global charged step size limit [cm]
    bool      enable_fluctuation{}; //!< Enable energy loss fluctuations

    // When fixed step limiter is used, this is the corresponding action ID
    ActionId fixed_step_action{};

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return max_particle_processes > 0 && model_to_action >= 3
               && num_models > 0 && scaling_min_range > 0
               && scaling_fraction > 0 && energy_fraction > 0
               && linear_loss_limit > 0
               && ((fixed_step_limiter > 0)
                   == static_cast<bool>(fixed_step_action));
    }

    //! Stop early due to range limitation
    CELER_FORCEINLINE_FUNCTION ActionId range_action() const
    {
        return ActionId{model_to_action - 3};
    }

    //! Undergo a discrete interaction
    CELER_FORCEINLINE_FUNCTION ActionId discrete_action() const
    {
        return ActionId{model_to_action - 2};
    }

    //! Indicate a discrete interaction was rejected by the integral method
    CELER_FORCEINLINE_FUNCTION ActionId integral_rejection_action() const
    {
        return ActionId{model_to_action - 1};
    }

    //! Indicate an interaction failed to allocate memory
    CELER_FORCEINLINE_FUNCTION ActionId failure_action() const
    {
        return ActionId{model_to_action + num_models};
    }
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
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ParticleItems = Collection<T, W, M, ParticleId>;

    //// DATA ////

    // Backend storage
    Items<real_type>            reals;
    Items<ModelId>              model_ids;
    Items<ValueGrid>            value_grids;
    Items<ValueGridId>          value_grid_ids;
    Items<ProcessId>            process_ids;
    Items<ValueTable>           value_tables;
    Items<IntegralXsProcess>    integral_xs;
    Items<ModelGroup>           model_groups;
    ParticleItems<ProcessGroup> process_groups;

    // Special data
    HardwiredModels<W, M> hardwired;
    FluctuationData<W, M> fluctuation;

    // Non-templated data
    PhysicsParamsScalars scalars;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !process_groups.empty()
               && (static_cast<bool>(fluctuation)
                   || !scalars.enable_fluctuation)
               && scalars;
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
        integral_xs    = other.integral_xs;
        model_groups   = other.model_groups;
        process_groups = other.process_groups;

        hardwired   = other.hardwired;
        fluctuation = other.fluctuation;

        scalars = other.scalars;

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
 * - Current macroscopic cross section
 */
struct PhysicsTrackState
{
    real_type interaction_mfp; //!< Remaining MFP to interaction
    real_type macro_xs; //!< Total cross section for discrete interactions
    real_type energy_deposition; //!< Local energy deposition in a step [MeV]
    Span<Secondary> secondaries; //!< Emitted secondaries

    // CURRENTLY UNUSED
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
    //// TYPES ////

    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    template<class T>
    using Items = celeritas::Collection<T, W, M>;

    //// DATA ////

    StateItems<PhysicsTrackState> state;    //!< Track state [track]
    StateItems<MscStep>           msc_step; //!< Internal MSC data [track]

    Items<real_type> per_process_xs; //!< XS [track][particle process]

    //// METHODS ////

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
        msc_step       = other.msc_step;
        per_process_xs = other.per_process_xs;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void resize(
    PhysicsStateData<Ownership::value, M>*                               state,
    const PhysicsParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type                                                            size)
{
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params.scalars.max_particle_processes > 0);
    make_builder(&state->state).resize(size);
    if (params.hardwired.msc)
    {
        make_builder(&state->msc_step).resize(size);
    }
    make_builder(&state->per_process_xs)
        .resize(size * params.scalars.max_particle_processes);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
