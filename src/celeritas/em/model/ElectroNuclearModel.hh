//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/ElectroNuclearModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/ElectroNuclearData.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

class EmExtraPhysicsHelper;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the electro-nuclear model interaction.
 */
class ElectroNuclearModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    using MevEnergy = units::MevEnergy;
    using HostRef = HostCRef<ElectroNuclearData>;
    using DeviceRef = DeviceCRef<ElectroNuclearData>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    ElectroNuclearModel(ActionId id,
                        ParticleParams const& particles,
                        MaterialParams const& materials);

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
    std::shared_ptr<EmExtraPhysicsHelper> helper_;

    // Host/device storage and reference
    CollectionMirror<ElectroNuclearData> data_;

    //// TYPES ////

    using HostXsData = HostVal<ElectroNuclearData>;

    //// HELPER FUNCTIONS ////

    inp::Grid
    calc_micro_xs(AtomicNumber atomic_number, double emin, double emax) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
