//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/NeutronInelasticModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/neutron/data/NeutronInelasticData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
struct CascadeOptions;
struct ImportPhysicsVector;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the neutron inelastic model interaction.
 */
class NeutronInelasticModel final : public Model, public ConcreteAction
{
  public:
    //!@{
    using AtomicMassNumber = IsotopeView::AtomicMassNumber;
    using MevEnergy = units::MevEnergy;
    using ReadData = std::function<ImportPhysicsVector(AtomicNumber)>;
    using HostRef = NeutronInelasticHostRef;
    using DeviceRef = NeutronInelasticDeviceRef;
    //!@}

  public:
    // Construct from model ID and other necessary data
    NeutronInelasticModel(ActionId id,
                          ParticleParams const& particles,
                          MaterialParams const& materials,
                          CascadeOptions const& options,
                          ReadData load_data);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    MicroXsBuilders micro_xs(Applicability) const final;

    //! Apply the interaction kernel to host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    HostRef const& host_ref() const { return data_.host_ref(); }
    DeviceRef const& device_ref() const { return data_.device_ref(); }
    //!@}

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<NeutronInelasticData> data_;

    //// TYPES ////

    using HostXsData = HostVal<NeutronInelasticData>;

    struct ChannelData
    {
        StepanovParameters par;
        Array<double, 13> xs;
        Array<double, 6 * 19> cdf;  //! [energy][angle]
    };

    //// HELPER FUNCTIONS ////

    Span<double const> get_xs_energy_bins() const;
    static ChannelData const& get_channel_data(ChannelId id);

    Span<double const> get_cdf_energy_bins() const;
    Span<double const> get_cos_theta_bins() const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
