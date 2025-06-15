//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/global/ActionInterface.hh"

#include "Model.hh"
#include "PhysicsData.hh"
#include "PhysicsOptions.hh"
#include "Process.hh"

namespace celeritas
{
class ActionRegistry;
class AtomicRelaxationParams;
class MaterialParams;
class ParticleParams;

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
 * - "failure": interactor failed to allocate secondaries
 */
class PhysicsParams final : public ParamsDataInterface<PhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstProcess = std::shared_ptr<Process const>;
    using SPConstModel = std::shared_ptr<Model const>;
    using SPConstRelaxation = std::shared_ptr<AtomicRelaxationParams const>;

    using VecProcess = std::vector<SPConstProcess>;
    using SpanConstProcessId = Span<ProcessId const>;
    using ActionIdRange = Range<ActionId>;
    using Options = PhysicsOptions;
    //!@}

    //! Physics parameter construction arguments
    struct Input
    {
        SPConstParticles particles;
        SPConstMaterials materials;
        VecProcess processes;
        SPConstRelaxation relaxation;  //!< Optional atomic relaxation
        ActionRegistry* action_registry = nullptr;

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
    inline SPConstModel const& model(ModelId) const;

    // Get a process
    inline SPConstProcess const& process(ProcessId) const;

    // Get the process for the given model
    inline ProcessId process_id(ModelId id) const;

    // Get the action IDs for all models
    inline ActionIdRange model_actions() const;

    // Get the processes that apply to a particular particle
    SpanConstProcessId processes(ParticleId) const;

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return host_ref_; }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return device_ref_; }

  private:
    using BC = SplineDerivCalculator::BoundaryCondition;
    using SPAction = std::shared_ptr<StaticConcreteAction>;
    using VecModel = std::vector<std::pair<SPConstModel, ProcessId>>;
    using HostValue = celeritas::HostVal<PhysicsParamsData>;
    using DeviceValue = PhysicsParamsData<Ownership::value, MemSpace::device>;

    // Kernels/actions
    SPAction pre_step_action_;
    SPAction msc_action_;
    SPAction range_action_;
    SPAction discrete_action_;
    SPAction integral_rejection_action_;
    SPAction failure_action_;
    SPAction fixed_step_action_;

    // Host metadata/access
    VecProcess processes_;
    VecModel models_;
    SPConstRelaxation relaxation_;

    // Host/device storage and reference
    HostValue host_;
    HostRef host_ref_;
    DeviceValue device_;
    DeviceRef device_ref_;

  private:
    VecModel build_models(ActionRegistry*) const;
    void build_options(Options const&, HostValue*) const;
    void build_particle_options(ParticleOptions const&, ParticleScalars*) const;
    void build_ids(ParticleParams const&, HostValue*) const;
    void build_tables(Options const&, MaterialParams const&, HostValue*) const;
    void build_model_tables(MaterialParams const&, HostValue*) const;
    void build_hardwired();
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
auto PhysicsParams::model(ModelId id) const -> SPConstModel const&
{
    CELER_EXPECT(id < this->num_models());
    return models_[id.get()].first;
}

//---------------------------------------------------------------------------//
/*!
 * Get a process.
 */
auto PhysicsParams::process(ProcessId id) const -> SPConstProcess const&
{
    CELER_EXPECT(id < this->num_processes());
    return processes_[id.get()];
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
}  // namespace celeritas
