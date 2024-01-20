//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/RayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/RayleighData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ImportedModelAdapter.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch Rayleigh scattering.
 */
class RayleighModel final : public Model
{
  public:
    //@{
    //! Type aliases
    using HostRef = HostCRef<RayleighData>;
    using DeviceRef = DeviceCRef<RayleighData>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //@}

  public:
    // Construct from model ID and other necessary data
    RayleighModel(ActionId id,
                  ParticleParams const& particles,
                  MaterialParams const& materials,
                  SPConstImported data);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    // Apply the interaction kernel to host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "scat-rayleigh"; }

    //! Short description of the post-step action
    std::string description() const final
    {
        return "interact by Rayleigh scattering";
    }

    //! Access Rayleigh data on the host
    HostRef const& host_ref() const { return mirror_.host_ref(); }

    //! Access Rayleigh data on the device
    DeviceRef const& device_ref() const { return mirror_.device_ref(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<RayleighData> mirror_;

    ImportedModelAdapter imported_;

    //// TYPES ////

    using HostValue = HostVal<RayleighData>;
    using ElScatParams = RayleighParameters;

    //// HELPER FUNCTIONS ////

    void build_data(HostValue* host_data, MaterialParams const& materials);
    static ElScatParams const& get_el_parameters(AtomicNumber atomic_number);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
