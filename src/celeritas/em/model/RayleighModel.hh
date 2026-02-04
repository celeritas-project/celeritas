//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/RayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/ParamsDataStore.hh"
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
class RayleighModel final : public Model, public StaticConcreteAction
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
    XsTable micro_xs(Applicability) const final;

    // Apply the interaction kernel to host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel to device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Access Rayleigh data on the host
    HostRef const& host_ref() const { return mirror_.host_ref(); }

    //! Access Rayleigh data on the device
    DeviceRef const& device_ref() const { return mirror_.device_ref(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    ParamsDataStore<RayleighData> mirror_;

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
