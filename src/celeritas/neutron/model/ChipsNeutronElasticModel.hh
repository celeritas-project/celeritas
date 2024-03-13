//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/ChipsNeutronElasticModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/neutron/data/NeutronElasticData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
struct ImportPhysicsVector;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the neutron elastic CHIPS model interaction.
 */
class ChipsNeutronElasticModel final : public Model
{
  public:
    //!@{
    using AtomicMassNumber = IsotopeView::AtomicMassNumber;
    using MevEnergy = units::MevEnergy;
    using ReadData = std::function<ImportPhysicsVector(AtomicNumber)>;
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
    MicroXsBuilders micro_xs(Applicability) const final;

    //! Apply the interaction kernel to host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "neutron-elastic-chips"; }

    //! Short description of the post-step action
    std::string description() const final
    {
        return "interact by neutron elastic scattering (CHIPS)";
    }

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
    void append_xs(ImportPhysicsVector const& inp, HostXsData* xs_data) const;
    void append_coeffs(AtomicMassNumber A, HostXsData* xs_data) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
