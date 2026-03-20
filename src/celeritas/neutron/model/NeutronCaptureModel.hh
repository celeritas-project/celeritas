//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/NeutronCaptureModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/ParamsDataStore.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/neutron/data/NeutronCaptureData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the neutron capture model interaction.
 */
class NeutronCaptureModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    using ReadData = std::function<inp::Grid(AtomicNumber)>;
    using HostRef = HostCRef<NeutronCaptureData>;
    using DeviceRef = DeviceCRef<NeutronCaptureData>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    NeutronCaptureModel(ActionId id,
                        ParticleParams const& particles,
                        MaterialParams const& materials,
                        ReadData load_data);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    XsTable micro_xs(Applicability) const final;

    //! Apply the interaction kernel to host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel to device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    HostRef const& host_ref() const { return mirror_.host_ref(); }
    DeviceRef const& device_ref() const { return mirror_.device_ref(); }
    //!@}

  private:
    //// DATA ////

    // Host/device storage and reference
    ParamsDataStore<NeutronCaptureData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
