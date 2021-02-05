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
#include "base/DeviceVector.hh"
#include "base/Types.hh"
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
    //!@}

    //! Global physics configuration options
    struct Options
    {
        real_type max_step_over_range; //!< alpha_r, limit step range at high E
        real_type min_step;            //!< rho_R, minimum range at low E
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
    size_type num_models() const { return models_.size(); }

    //! Number of processes
    size_type num_processes() const { return processes_.size(); }

    //! Number of particle types
    size_type num_particles() const { return processes_by_particle_.size(); }

    //! Maximum number of processes that apply to any one particle
    size_type max_particle_processes() const { return max_pbp_; }

    // Get a model
    inline const Model& model(ModelId) const;

    // Get a process
    inline const Process& process(ProcessId) const;

    // Get the processes that apply to a particlular particle
    inline SpanConstProcessId processes(ParticleId) const;

    //// DEVICE ACCESSORS ////

    // Get managed data
    PhysicsParamsPointers device_pointers() const;

  private:
    using SPConstModel = std::shared_ptr<const Model>;

    //// HOST DATA ////

    VecProcess                          processes_;
    std::vector<std::vector<ProcessId>> processes_by_particle_;
    std::vector<SPConstModel>           models_;
    size_type                           max_pbp_ = 0;

    //// DEVICE DATA ////

    DeviceVector<UniformGridData>     grid_inputs_;
    DeviceVector<real_type>           xsgrid_data_;
    DeviceVector<XsGridPointers>      value_grids_;
    DeviceVector<ValueTable>          value_tables_;
    DeviceVector<ProcessId>           process_ids_;
    DeviceVector<ModelId>             model_ids_;
    DeviceVector<real_type>           model_energy_;
    DeviceVector<ModelGroup>          model_groups_;
    DeviceVector<ProcessGroup>        process_groups_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a model.
 */
const Model& PhysicsParams::model(ModelId id) const
{
    CELER_EXPECT(id < this->num_models());
    return *models_[id.get()];
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
 * Get the list of process IDs that apply to a particle type.
 */
auto PhysicsParams::processes(ParticleId id) const -> SpanConstProcessId
{
    CELER_EXPECT(id < this->num_processes());
    return make_span(processes_by_particle_[id.get()]);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
