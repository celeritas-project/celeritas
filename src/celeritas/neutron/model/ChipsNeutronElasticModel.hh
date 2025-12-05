//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/ChipsNeutronElasticModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/neutron/data/NeutronElasticData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the neutron elastic CHIPS model interaction.
 */
class ChipsNeutronElasticModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    using AtomicMassNumber = IsotopeView::AtomicMassNumber;
    using MevEnergy = units::MevEnergy;
    using ReadData = std::function<inp::Grid(AtomicNumber)>;
    using HostRef = NeutronElasticHostRef;
    using DeviceRef = NeutronElasticDeviceRef;
    //!@}

  public:
    // Construct from model ID and other necessary data
    ChipsNeutronElasticModel(ActionId id,
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
    CollectionMirror<NeutronElasticData> mirror_;

    //// TYPES ////

    using HostXsData = HostVal<NeutronElasticData>;

    //// HELPER FUNCTIONS ////
    void append_coeffs(AtomicMassNumber A, HostXsData* xs_data) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
