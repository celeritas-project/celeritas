//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/NeutronInelasticModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/neutron/data/NeutronInelasticData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
struct CascadeOptions;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the neutron inelastic model interaction.
 *
 * Only neutron-neutron (proton-proton) and neutron-proton channels are
 * tabulated in [10, 320] (MeV) where pion production is not likely. The cross
 * sections below 10 MeV will be calculated on the fly using the Stepanov's
 * function. Tabulated data of cross sections and parameters at the low energy
 * are from G4CascadePPChannel, G4CascadeNPChannel and G4CascadeNNChannel of
 * the Geant4 11.2 release while angular c.d.f data are from G4PP2PPAngDst and
 * G4NP2NPAngDst. Also note that the channel cross sections of nucleon-nucleon
 * are same as their total cross sections in the energy range and the
 * proton-proton channel is same as the neutron-neutron channel based on the
 * charge-independence hypothesis of the nuclear force.
 * See \cite{bertini-1963,hess-1958}.
 */
class NeutronInelasticModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    using AtomicMassNumber = IsotopeView::AtomicMassNumber;
    using MevEnergy = units::MevEnergy;
    using ReadData = std::function<inp::Grid(AtomicNumber)>;
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
    XsTable micro_xs(Applicability) const final;

    //! Apply the interaction kernel to host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel to device data
    void step(CoreParams const&, CoreStateDevice&) const final;

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

    //// HELPER FUNCTIONS ////

    StepanovParameters const& get_channel_params(ChannelId id);
    inp::Grid const& get_channel_xs(ChannelId id);
    inp::TwodGrid const& get_channel_cdf(ChannelId id);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
