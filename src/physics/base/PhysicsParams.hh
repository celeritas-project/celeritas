//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include "base/PieMirror.hh"
#include "base/Types.hh"
#include "base/Units.hh"
#include "Model.hh"
#include "Process.hh"
#include "PhysicsInterface.hh"
#include "Types.hh"

namespace celeritas
{
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
 * \c ModelId values, as well as the tables of cross section data.
 *
 * Input options are:
 * - \c min_range: below this value, there is no extra transformation from
 *   particle range to step length.
 * - \c max_step_over_range: at higher energy (longer range), gradually
 *   decrease the maximum step length until it's this fraction of the tabulated
 *   range.
 */
class PhysicsParams
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles   = std::shared_ptr<const ParticleParams>;
    using SPConstMaterials   = std::shared_ptr<const MaterialParams>;
    using SPConstProcess     = std::shared_ptr<const Process>;
    using VecProcess         = std::vector<SPConstProcess>;
    using SpanConstProcessId = Span<const ProcessId>;
    using HostRef
        = PhysicsParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = PhysicsParamsData<Ownership::const_reference, MemSpace::device>;
    //!@}

    //! Global physics configuration options
    struct Options
    {
        real_type min_range           = 1 * units::millimeter; //!< rho_R
        real_type max_step_over_range = 0.2;                   //!< alpha_r
    };

    //! Physics parameter construction arguments
    struct Input
    {
        SPConstParticles particles;
        SPConstMaterials materials;
        VecProcess       processes;

        Options options;
    };

  public:
    // Construct with processes and helper classes
    explicit PhysicsParams(Input);

    //// HOST ACCESSORS ////

    //! Number of models
    ModelId::value_type num_models() const { return models_.size(); }

    //! Number of processes
    ProcessId::value_type num_processes() const { return processes_.size(); }

    // Number of particle types
    inline ParticleId::value_type num_particles() const;

    // Maximum number of processes that apply to any one particle
    inline ProcessId::value_type max_particle_processes() const;

    // Get a model
    inline const Model& model(ModelId) const;

    // Get a process
    inline const Process& process(ProcessId) const;

    // Get the processes that apply to a particular particle
    SpanConstProcessId processes(ParticleId) const;

    //! Access material properties on the host
    const HostRef& host_pointers() const { return data_.host(); }

    //! Access material properties on the device
    const DeviceRef& device_pointers() const { return data_.device(); }

  private:
    using SPConstModel = std::shared_ptr<const Model>;
    using VecModel     = std::vector<std::pair<SPConstModel, ProcessId>>;
    using HostValue    = PhysicsParamsData<Ownership::value, MemSpace::host>;

    // Host metadata/access
    VecProcess processes_;
    VecModel   models_;

    // Host/device storage and reference
    PieMirror<PhysicsParamsData> data_;

  private:
    VecModel build_models() const;
    void     build_options(const Options& opts, HostValue* data) const;
    void     build_ids(const ParticleParams& particles, HostValue* data) const;
    void     build_xs(const MaterialParams& mats, HostValue* data) const;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Number of particle types.
 */
auto PhysicsParams::num_particles() const -> ParticleId::value_type
{
    return this->host_pointers().process_ids.size();
}

//---------------------------------------------------------------------------//
/*!
 * Number of particle types.
 */
auto PhysicsParams::max_particle_processes() const -> ProcessId::value_type
{
    return this->host_pointers().max_particle_processes;
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
} // namespace celeritas
