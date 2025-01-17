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
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/global/ActionInterface.hh"

#include "Model.hh"
#include "PhysicsData.hh"
#include "Process.hh"

namespace celeritas
{
class ActionRegistry;
class AtomicRelaxationParams;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Particle-dependent physics configuration options.
 *
 * These parameters have different values for light particles
 * (electrons/positrons) and heavy particles (muons/hadrons).
 *
 * Input options are:
 * - \c min_range: below this value, there is no extra transformation from
 *   particle range to step length.
 * - \c max_step_over_range: at higher energy (longer range), gradually
 *   decrease the maximum step length until it's this fraction of the tabulated
 *   range.
 * - \c lowest_energy: tracking cutoff kinetic energy.
 * - \c displaced: whether MSC lateral displacement is enabled for e-/e+
 * - \c range_factor: used in the MSC step limitation algorithm to restrict the
 *   step size to \f$ f_r \cdot max(r, \lambda) \f$ at the start of a track or
 *   after entering a volume, where \f$ f_r \f$ is the range factor, \f$ r \f$
 *   is the range, and \f$ \lambda \f$ is the mean free path.
 * - \c step_limit_algorithm: algorithm used to determine the MSC step limit.
 *
 * NOTE: min_range/max_step_over_range are not accessible through Geant4, and
 * they can also be set to be different for electrons, mu/hadrons, and ions
 * (they are set in Geant4 with \c G4EmParameters::SetStepFunction()).
 */
struct ParticleOptions
{
    using Energy = units::MevEnergy;

    //!@{
    //! \name Range calculation
    real_type min_range;
    real_type max_step_over_range;
    //!@}

    //!@{
    //! \name Energy loss
    Energy lowest_energy;
    //!@}

    //!@{
    //! \name Multiple scattering
    bool displaced;
    real_type range_factor;
    MscStepLimitAlgorithm step_limit_algorithm;
    //!@}

    //! Default options for light particles (electrons/positrons)
    static ParticleOptions default_light()
    {
        ParticleOptions opts;
        opts.min_range = real_type{1} * units::millimeter;
        opts.max_step_over_range = 0.2;
        opts.lowest_energy = ParticleOptions::Energy{0.001};
        opts.displaced = true;
        opts.range_factor = 0.04;
        opts.step_limit_algorithm = MscStepLimitAlgorithm::safety;
        return opts;
    };

    //! Default options for heavy particles (muons/hadrons)
    static ParticleOptions default_heavy()
    {
        ParticleOptions opts;
        opts.min_range = 0.1 * units::millimeter;
        opts.max_step_over_range = 0.2;
        opts.lowest_energy = ParticleOptions::Energy{0.001};
        opts.displaced = false;
        opts.range_factor = 0.2;
        opts.step_limit_algorithm = MscStepLimitAlgorithm::minimal;
        return opts;
    };
};

//---------------------------------------------------------------------------//
/*!
 * Physics configuration options.
 *
 * Input options are:
 * - \c fixed_step_limiter: if nonzero, prevent any tracks from taking a step
 *   longer than this length.
 * - \c min_eprime_over_e: energy scaling fraction used to estimate the maximum
 *   cross section over the step in the integral approach for energy loss
 *   processes.
 * - \c linear_loss_limit: if the mean energy loss along a step is greater than
 *   this fractional value of the pre-step kinetic energy, recalculate the
 *   energy loss.
 * - \c lambda_limit: limit on the MSC mean free path.
 * - \c safety_factor: used in the MSC step limitation algorithm to restrict
 *   the step size to \f$ f_s s \f$, where \f$ f_s \f$ is the safety factor and
 *   \f$  s \f$ is the safety distance.
 * - \c secondary_stack_factor: the number of secondary slots per track slot
 *   allocated.
 * - \c disable_integral_xs: for particles with energy loss processes, the
 *   particle energy changes over the step, so the assumption that the cross
 *   section is constant is no longer valid. By default, many charged particle
 *   processes use MC integration to sample the discrete interaction length
 *   with the correct probability. Disable this integral approach for all
 *   processes.
 *   \c spline_eloss_order: the order of interpolation to be used for the
 *   spline interpolation. If it is 1, then the existing linear interpolation
 *   is used. If it is 2+, the spline interpolation is used for energy loss
 *   using the specified order. Default value is 1.
 */
struct PhysicsParamsOptions
{
    //!@{
    //! \name Range calculation
    real_type fixed_step_limiter{0};
    //!@}

    //!@{
    //! \name Energy loss
    real_type min_eprime_over_e{0.8};
    real_type linear_loss_limit{0.01};
    //!@}

    //!@{
    //! \name Multiple scattering
    real_type lambda_limit{real_type{1} * units::millimeter};
    real_type safety_factor{0.6};
    //!@}

    //!@{
    //! \name Particle-dependent parameters
    ParticleOptions light{ParticleOptions::default_light()};
    ParticleOptions heavy{ParticleOptions::default_heavy()};
    //!@}

    real_type secondary_stack_factor{3};
    bool disable_integral_xs{false};
    size_type spline_eloss_order{1};
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
    using Options = PhysicsParamsOptions;
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
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    using SPAction = std::shared_ptr<StaticConcreteAction>;
    using VecModel = std::vector<std::pair<SPConstModel, ProcessId>>;
    using HostValue = celeritas::HostVal<PhysicsParamsData>;

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
    CollectionMirror<PhysicsParamsData> data_;

  private:
    VecModel build_models(ActionRegistry*) const;
    void build_options(Options const& opts, HostValue* data) const;
    void build_particle_options(ParticleOptions const&, ParticleScalars*) const;
    void build_ids(ParticleParams const& particles, HostValue* data) const;
    void build_xs(Options const& opts,
                  MaterialParams const& mats,
                  HostValue* data) const;
    void build_model_xs(MaterialParams const& mats, HostValue* data) const;
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
