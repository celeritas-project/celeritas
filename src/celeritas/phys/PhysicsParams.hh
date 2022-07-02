//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/global/ActionInterface.hh"

#include "Model.hh"
#include "PhysicsData.hh"
#include "Process.hh"

namespace celeritas
{
class ActionManager;
class AtomicRelaxationParams;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Physics configuration options.
 *
 * Input options are:
 * - \c min_range: below this value, there is no extra transformation from
 *   particle range to step length.
 * - \c max_step_over_range: at higher energy (longer range), gradually
 *   decrease the maximum step length until it's this fraction of the tabulated
 *   range.
 * - \c min_eprime_over_e: energy scaling fraction used to estimate the maximum
 *   cross section over the step in the integral approach for energy loss
 *   processes.
 * - \c linear_loss_limit: if the mean energy loss along a step is greater than
 *   this fractional value of the pre-step kinetic energy, recalculate the
 *   energy loss.
 * - \c secondary_stack_factor: the number of secondary slots per track slot
 *   allocated.
 * - \c disable_integral_xs: for particles with energy loss processes, the
 *   particle energy changes over the step, so the assumption that the cross
 *   section is constant is no longer valid. By default, many charged particle
 *   processes use MC integration to sample the discrete interaction length
 *   with the correct probability. Disable this integral approach for all
 *   processes.
 * - \c enable_fluctuation: enable simulation of energy loss fluctuations.
 */
struct PhysicsParamsOptions
{
    real_type min_range              = 1 * units::millimeter;
    real_type max_step_over_range    = 0.2;
    real_type min_eprime_over_e      = 0.8;
    real_type fixed_step_limiter     = 0;
    real_type linear_loss_limit      = 0.01;
    real_type secondary_stack_factor = 3;
    bool      disable_integral_xs    = false;
    bool      enable_fluctuation     = true;
};

//---------------------------------------------------------------------------//
/*!
 * Manage physics processes and models.
 *
 * The physics params takes a vector of processes and sets up the processes and
 * models. It constructs data and mappings of data:
 * - particle type and process to tabulated values of cross sections etc,
 * - particle type to applicable processes
 *
 * During construction it constructs models and their corresponding list of
 * \c ActionId values, as well as the tables of cross section data. Besides the
 * individual interaction kernels, the physics parameters manage additional
 * actions:
 * - "pre-step": calculate physics step limits
 * - "along-step": propagate, apply energy loss, multiple scatter
 * - "range": limit step by energy loss
 * - "discrete-select": sample a process for a discrete interaction, or reject
 *   due to integral cross sectionl
 * - "integral-rejected": do not apply a discrete interaction
 * - "failure": model failed to allocate secondaries
 */
class PhysicsParams
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles  = std::shared_ptr<const ParticleParams>;
    using SPConstMaterials  = std::shared_ptr<const MaterialParams>;
    using SPConstProcess    = std::shared_ptr<const Process>;
    using SPConstRelaxation = std::shared_ptr<const AtomicRelaxationParams>;

    using VecProcess         = std::vector<SPConstProcess>;
    using SpanConstProcessId = Span<const ProcessId>;
    using ActionIdRange      = Range<ActionId>;
    using Options            = PhysicsParamsOptions;

    using HostRef
        = PhysicsParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = PhysicsParamsData<Ownership::const_reference, MemSpace::device>;
    //!@}

    //! Physics parameter construction arguments
    struct Input
    {
        SPConstParticles  particles;
        SPConstMaterials  materials;
        VecProcess        processes;
        SPConstRelaxation relaxation; //!< Optional atomic relaxation
        ActionManager*    action_manager = nullptr;

        Options options;
    };

  public:
    // Construct with processes and helper classes
    explicit PhysicsParams(Input);

    //// HOST ACCESSORS ////

    //! Number of models
    ModelId::size_type num_models() const { return models_.size(); }

    //! Number of processes
    ProcessId::size_type num_processes() const { return processes_.size(); }

    // Number of particle types
    inline ParticleId::size_type num_particles() const;

    // Maximum number of processes that apply to any one particle
    inline ProcessId::size_type max_particle_processes() const;

    // Get a model
    inline const Model& model(ModelId) const;

    // Get a process
    inline const Process& process(ProcessId) const;

    // Get the process for the given model
    inline ProcessId process_id(ModelId id) const;

    // Get the action IDs for all models
    inline ActionIdRange model_actions() const;

    // Get the processes that apply to a particular particle
    SpanConstProcessId processes(ParticleId) const;

    //! Access physics properties on the host
    const HostRef& host_ref() const { return data_.host(); }

    //! Access physics properties on the device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    using SPConstModel = std::shared_ptr<const Model>;
    using SPAction     = std::shared_ptr<ConcreteAction>;
    using VecModel     = std::vector<std::pair<SPConstModel, ProcessId>>;
    using HostValue    = PhysicsParamsData<Ownership::value, MemSpace::host>;

    // Kernels/actions
    SPAction pre_step_action_;
    SPAction along_step_action_;
    SPAction range_action_;
    SPAction discrete_action_;
    SPAction integral_rejection_action_;
    SPAction failure_action_;
    SPAction fixed_step_action_;

    // Host metadata/access
    VecProcess        processes_;
    VecModel          models_;
    SPConstRelaxation relaxation_;

    // Host/device storage and reference
    CollectionMirror<PhysicsParamsData> data_;

  private:
    VecModel build_models(ActionManager*) const;
    void     build_options(const Options& opts, HostValue* data) const;
    void     build_ids(const ParticleParams& particles, HostValue* data) const;
    void     build_xs(const Options&        opts,
                      const MaterialParams& mats,
                      HostValue*            data) const;
    void     build_model_xs(const MaterialParams& mats, HostValue* data) const;
    void     build_fluct(const Options&        opts,
                         const MaterialParams& mats,
                         const ParticleParams& particles,
                         HostValue*            data) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Number of particle types.
 */
auto PhysicsParams::num_particles() const -> ParticleId::size_type
{
    return this->host_ref().process_ids.size();
}

//---------------------------------------------------------------------------//
/*!
 * Number of particle types.
 */
auto PhysicsParams::max_particle_processes() const -> ProcessId::size_type
{
    return this->host_ref().scalars.max_particle_processes;
}

//---------------------------------------------------------------------------//
/*!
 * Get a model.
 */
const Model& PhysicsParams::model(ModelId id) const
{
    CELER_EXPECT(id < this->num_models());
    return *models_[id.get()].first;
}

//---------------------------------------------------------------------------//
/*!
 * Get a process.
 */
const Process& PhysicsParams::process(ProcessId id) const
{
    CELER_EXPECT(id < this->num_processes());
    return *processes_[id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the process ID of the given model.
 */
ProcessId PhysicsParams::process_id(ModelId id) const
{
    CELER_EXPECT(id < this->num_models());
    return models_[id.get()].second;
}

//---------------------------------------------------------------------------//
/*!
 * Get the action kernel IDs for all models.
 */
auto PhysicsParams::model_actions() const -> ActionIdRange
{
    auto offset = host_ref().scalars.model_to_action;
    return {ActionId{offset}, ActionId{offset + this->num_models()}};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
