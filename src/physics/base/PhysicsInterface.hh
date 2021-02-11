//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "Types.hh"
#include "physics/grid/XsGridInterface.hh"
#include "physics/em/detail/LivermorePE.hh"
#include "physics/em/detail/EPlusGG.hh"
#include "physics/material/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Currently all value grids are cross section grids
using ValueGrid = XsGridData;

//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
//! Hardcoded types of grid data
enum class PhysicsTableType
{
    macro_xs,    //!< Interaction cross sections
    energy_loss, //!< Energy loss per unit length
    range,       //!< Particle range
    size_        //!< Sentinel value
};

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
    Span<const real_type> energy; //!< Energy grid bounds [MeV]
    Span<const ModelId>   model;  //!< Corresponding models

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
    Span<const ValueGrid> material; //!< Value grid by material index

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !material.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Processes for a single particle type.
 *
 * Each index should be accessed with type ParticleProcessId. The "tables" are
 * a fixed-size number of Span references to ValueTables. The first index of
 * the table (hard-coded) corresponds to PhysicsTableType; the second index is
 * a ParticleProcessId. So the cross sections for ParticleProcessId{2} would
 * be \code tables[size_type(PhysicsTableType::macro_xs)][2] \endcode. This
 * awkward access is encapsulated by the PhysicsTrackView.
 */
struct ProcessGroup
{
    Span<const ProcessId> processes; //!< Processes that apply

    Array<Span<const ValueTable>, size_type(PhysicsTableType::size_)> tables; //!< Data
    Span<const ModelGroup> models; //!< Model applicability

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !processes.empty() && models.size() == processes.size();
    }

    //! Number of processes that apply
    CELER_FUNCTION ParticleProcessId::value_type size() const
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
    ProcessId                          gamma_photoelectric;
    const detail::LivermorePEPointers* livermore_params = nullptr;
    ProcessId                          positron_annihilation;
    const detail::EPlusGGPointers*     eplusgg_params = nullptr;
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
 params.particle[1].table[int(PhysicsTableType::macro_xs)][0].material[2].log_energy;
 * \endcode
 */
struct PhysicsParamsPointers
{
    Span<const ProcessGroup> particle;
    HardwiredModels          hardwired;
    size_type                max_particle_processes{};

    //// USER-CONFIGURABLE CONSTANTS ////
    real_type scaling_min_range{}; //!< rho [cm]
    real_type scaling_fraction{};  //!< alpha [unitless]
    // real_type max_eloss_fraction{};  //!< For scaled range calculation

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !particle.empty() && max_particle_processes;
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
    real_type          interaction_mfp; //!< Remaining MFP to interaction
    real_type          step_length;     //!< Maximum step length
    real_type          macro_xs;
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
struct PhysicsStatePointers
{
    Span<PhysicsTrackState> state;          //!< Track state [track]
    Span<real_type>         per_process_xs; //!< XS [track][particle process]

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
