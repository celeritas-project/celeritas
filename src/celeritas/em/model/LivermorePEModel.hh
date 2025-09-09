//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/LivermorePEModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
struct ImportLivermorePE;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the Livermore photoelectric model interaction.
 */
class LivermorePEModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    using ReadData = std::function<ImportLivermorePE(AtomicNumber)>;
    using HostRef = HostCRef<LivermorePEData>;
    using DeviceRef = DeviceCRef<LivermorePEData>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    LivermorePEModel(ActionId id,
                     ParticleParams const& particles,
                     MaterialParams const& materials,
                     ReadData load_data);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    XsTable micro_xs(Applicability) const final;

    // Apply the interaction kernel on host
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel on device
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Access data on the host
    HostRef const& host_ref() const { return data_.host_ref(); }

    //! Access data on the device
    DeviceRef const& device_ref() const { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<LivermorePEData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
