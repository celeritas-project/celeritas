//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/StackAllocatorData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/AtomicRelaxationData.hh"
#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/neutron/data/NeutronElasticData.hh"

#include "Interaction.hh"
#include "Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Set of value grids for all elements or materials.
 *
 * It is allowable for this to be "false" (i.e. no materials assigned)
 * indicating that the value table doesn't apply in the context -- for
 * example, an empty ValueTable macro_xs means that the process doesn't have a
 * discrete interaction.
 */
template<class GridId>
struct ValueTable
{
    ItemRange<GridId> grids;  //!< Value grid by element or material index

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !grids.empty(); }
};

using UniformTable = ValueTable<UniformGridId>;

//---------------------------------------------------------------------------//
/*!
 * Set of cross section CDF tables for a model.
 *
 * Each material has a set of value grids for its constituent elements; these
 * are used to sample an element from a material when required by a discrete
 * interaction. A null \c ValueTableId means the material only has a single
 * element, so no cross sections need to be stored. An empty \c ModelCdfTable
 * means no element selection is required for the model.
 */
struct ModelCdfTable
{
    ItemRange<UniformTable> tables;  //!< Value table by material

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !tables.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Energy-dependent model IDs for a single process and particle type.
 *
 * For a given particle type, a single process should be divided into multiple
 * models as a function of energy. The \c ModelGroup represents this with an
 * energy grid, and each cell of the grid corresponding to a particular
 * \c ParticleModelId.
 */
struct ModelGroup
{
    using Energy = units::MevEnergy;

    ItemRange<real_type> energy;  //!< Energy grid bounds [MeV]
    ItemRange<ParticleModelId> model;  //!< Corresponding models

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return (energy.size() >= 2) && (model.size() + 1 == energy.size());
    }
};

//---------------------------------------------------------------------------//
/*!
 * Particle-process that uses MC integration to sample interaction length.
 *
 * This is needed for the integral approach for correctly sampling the discrete
 * interaction length after a particle loses energy along a step. An \c
 * IntegralXsProcess is stored for each particle-process. This will be "false"
 * (i.e. no energy_max assigned) if the particle associated with the process
 * does not have energy loss processes or if \c use_integral_xs is false.
 */
struct IntegralXsProcess
{
    ItemRange<real_type> energy_max_xs;  //!< Energy of the largest xs [mat]

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
 * Each index should be accessed with type ParticleProcessId. \c macro_xs
 * stores the cross section tables for each process, while \c energy_loss and
 * \c range are the process-integrated dE/dx and range for the particle.  \c
 * integral_xs will only be assigned if the integral approach is used and the
 * particle has continuous-discrete processes.
 *
 * \todo If it's possible for a particle to have multiple at-rest processes, \c
 * at_rest should be the process with the smallest lifetime. This is used in \c
 * select_discrete_interaction to choose the at-rest process with the smallest
 * time to interaction if the particle is stopped and has a process that
 * applies at rest.
 */
struct ProcessGroup
{
    ItemRange<ProcessId> processes;  //!< Processes that apply [ppid]
    ItemRange<ModelGroup> models;  //!< Model applicability [ppid]
    ItemRange<IntegralXsProcess> integral_xs;  //!< [ppid]
    ItemRange<ValueTable<XsGridId>> macro_xs;  //!< [ppid]
    UniformTable energy_loss;  //!< Process-integrated energy loss
    UniformTable range;  //!< Process-integrated range
    UniformTable inverse_range;  //!< Inverse process-integrated range
    ParticleProcessId at_rest;  //!< ID of the particle's at-rest process

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
 * IDs for models that do on-the-fly cross section calculation.
 */
struct HardwiredIds
{
    ProcessId annihilation;
    ModelId eplusgg;

    ProcessId photoelectric;
    ModelId livermore_pe;

    ProcessId neutron_elastic;
    ModelId chips;
};

//---------------------------------------------------------------------------//
/*!
 * Model data for special hardwired cases (on-the-fly xs calculations).
 */
template<Ownership W, MemSpace M>
struct HardwiredModels
{
    // Process and model IDs
    HardwiredIds ids;

    // Model data
    EPlusGGData eplusgg;
    LivermorePEData<W, M> livermore_pe;
    AtomicRelaxParamsData<W, M> relaxation;
    NeutronElasticData<W, M> chips;

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    HardwiredModels& operator=(HardwiredModels<W2, M2> const& other)
    {
        // Don't assign the references to model data
        ids = other.ids;
        eplusgg = other.eplusgg;

        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * User-configurable particle-dependent physics constants.
 *
 * These scalar quantities can have different values for electrons/positrons
 * and muons/hadrons. They are described in \c PhysicsParams .
 */
struct ParticleScalars
{
    using Energy = units::MevEnergy;

    // Energy loss/range options
    real_type min_range{};  //!< rho [len]
    real_type max_step_over_range{};  //!< alpha [unitless]
    Energy lowest_energy{};  //!< Lowest kinetic energy

    // Multiple scattering options
    bool displaced{};  //!< Whether lateral displacement is enabled
    real_type range_factor{};
    MscStepLimitAlgorithm step_limit_algorithm{MscStepLimitAlgorithm::size_};

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return min_range > 0 && max_step_over_range > 0
               && lowest_energy > zero_quantity() && range_factor > 0
               && range_factor < 1
               && step_limit_algorithm != MscStepLimitAlgorithm::size_;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Scalar (no template needed) quantities used by physics.
 *
 * The user-configurable constants and multiple scattering options are
 * described in \c PhysicsParams .
 *
 * The \c first_model_action is the \c ActionId for the first \c
 * ModelId . Additionally it implies (by construction in physics_params) the
 * action IDs of several other physics actions.
 */
struct PhysicsParamsScalars
{
    using Energy = units::MevEnergy;

    //! Highest number of processes for any particle type
    ProcessId::size_type max_particle_processes{};
    //! Offset to create an ActionId from a ModelId
    ActionId first_model_action{};
    //! Number of physics models
    ModelId::size_type num_models{};

    // User-configurable constants
    real_type min_eprime_over_e{};  //!< xi [unitless]
    real_type linear_loss_limit{};  //!< For scaled range calculation
    real_type fixed_step_limiter{};  //!< Global charged step size limit [len]

    // User-configurable multiple scattering options
    real_type lambda_limit{};  //!< lambda limit
    real_type safety_factor{};  //!< safety factor

    // Particle-dependent user-configurable constants
    ParticleScalars light;
    ParticleScalars heavy;

    real_type secondary_stack_factor = 3;  //!< Secondary storage per state
                                           //!< size
    // When fixed step limiter is used, this is the corresponding action ID
    ActionId fixed_step_action{};

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return max_particle_processes > 0 && first_model_action >= ActionId{4}
               && num_models > 0 && min_eprime_over_e > 0
               && linear_loss_limit > 0 && secondary_stack_factor > 0
               && ((fixed_step_limiter > 0)
                   == static_cast<bool>(fixed_step_action))
               && lambda_limit > 0 && safety_factor >= 0.1 && light && heavy;
    }

    //! Stop early due to MSC limitation
    CELER_FORCEINLINE_FUNCTION ActionId msc_action() const
    {
        return first_model_action - 4;
    }

    //! Stop early due to range limitation
    CELER_FORCEINLINE_FUNCTION ActionId range_action() const
    {
        return first_model_action - 3;
    }

    //! Undergo a discrete interaction
    CELER_FORCEINLINE_FUNCTION ActionId discrete_action() const
    {
        return first_model_action - 2;
    }

    //! Indicate a discrete interaction was rejected by the integral method
    CELER_FORCEINLINE_FUNCTION ActionId integral_rejection_action() const
    {
        return first_model_action - 1;
    }

    //! Indicate an interaction failed to allocate memory
    CELER_FORCEINLINE_FUNCTION ActionId failure_action() const
    {
        return first_model_action + num_models;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent shared physics data.
 *
 * This includes macroscopic cross section tables ordered by
 * [particle][process][material][energy] and process-integrated energy loss and
 * range tables ordered by [particle][material][energy].
 */
template<Ownership W, MemSpace M>
struct PhysicsParamsData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ParticleItems = Collection<T, W, M, ParticleId>;
    template<class T>
    using ParticleModelItems = Collection<T, W, M, ParticleModelId>;

    //// DATA ////

    // Non-templated data
    PhysicsParamsScalars scalars;

    // Models that calculate cross sections on the fly
    HardwiredModels<Ownership::const_reference, M> hardwired;

    // Grid and table storage
    Items<XsGridId> xs_grid_ids;
    Items<XsGridRecord> xs_grids;
    Items<ValueTable<XsGridId>> xs_tables;
    Items<UniformGridId> uniform_grid_ids;
    Items<UniformGridRecord> uniform_grids;
    Items<UniformTable> uniform_tables;
    ParticleModelItems<ModelCdfTable> model_cdf;

    // Process and model storage
    Items<ModelGroup> model_groups;
    Items<IntegralXsProcess> integral_xs;
    ParticleItems<ProcessGroup> process_groups;
    ParticleModelItems<ModelId> model_ids;
    Items<ParticleModelId> pmodel_ids;
    Items<ProcessId> process_ids;

    // Backend storage
    Items<real_type> reals;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !process_groups.empty() && !model_ids.empty() && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    PhysicsParamsData& operator=(PhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        scalars = other.scalars;

        hardwired = other.hardwired;

        xs_grids = other.xs_grids;
        xs_grid_ids = other.xs_grid_ids;
        xs_tables = other.xs_tables;
        uniform_grids = other.uniform_grids;
        uniform_grid_ids = other.uniform_grid_ids;
        uniform_tables = other.uniform_tables;
        model_cdf = other.model_cdf;

        model_groups = other.model_groups;
        integral_xs = other.integral_xs;
        process_groups = other.process_groups;
        model_ids = other.model_ids;
        pmodel_ids = other.pmodel_ids;
        process_ids = other.process_ids;

        reals = other.reals;

        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Physics state data for a single track.
 *
 * State that's persistent across steps:
 * - Remaining number of mean free paths to the next discrete interaction
 *
 * State that is reset at every step:
 * - Current macroscopic cross section
 * - Within-step energy deposition
 * - Within-step energy loss range
 * - Secondaries emitted from an interaction
 * - Discrete process element selection
 */
struct PhysicsTrackState
{
    real_type interaction_mfp;  //!< Remaining MFP to interaction

    // TEMPORARY STATE
    real_type macro_xs;  //!< Total cross section for discrete interactions
    real_type energy_deposition;  //!< Local energy deposition in a step [MeV]
    real_type dedx_range;  //!< Local energy loss range [len]
    MscRange msc_range;  //!< Range properties for multiple scattering
    Span<Secondary> secondaries;  //!< Emitted secondaries
    ElementComponentId element;  //!< Element sampled for interaction
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

    StateItems<PhysicsTrackState> state;  //!< Track state [track]
    StateItems<MscStep> msc_step;  //!< Internal MSC data [track]

    Items<real_type> per_process_xs;  //!< XS [track][particle process]

    AtomicRelaxStateData<W, M> relaxation;  //!< Scratch data
    StackAllocatorData<Secondary, W, M> secondaries;  //!< Secondary stack

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !state.empty() && secondaries;
    }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    PhysicsStateData& operator=(PhysicsStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        state = other.state;
        msc_step = other.msc_step;

        per_process_xs = other.per_process_xs;

        relaxation = other.relaxation;
        secondaries = other.secondaries;

        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void resize(PhysicsStateData<Ownership::value, M>* state,
                   HostCRef<PhysicsParamsData> const& params,
                   size_type size)
{
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params.scalars.max_particle_processes > 0);
    resize(&state->state, size);
    resize(&state->msc_step, size);
    resize(&state->per_process_xs,
           size * params.scalars.max_particle_processes);
    resize(&state->relaxation, params.hardwired.relaxation, size);
    resize(
        &state->secondaries,
        static_cast<size_type>(size * params.scalars.secondary_stack_factor));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
