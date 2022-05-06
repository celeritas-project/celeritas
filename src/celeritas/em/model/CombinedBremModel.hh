//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/CombinedBremModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/CombinedBremData.hh"
#include "celeritas/phys/Model.hh"

#include "RelativisticBremModel.hh"
#include "SeltzerBergerModel.hh"

namespace celeritas
{
struct ImportSBTable;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch a combined model of SeltzerBergerModel at the low energy
 * and RelativisticBremModel at the hight energy for e+/e- Bremsstrahlung.
 */
class CombinedBremModel final : public Model
{
  public:
    //@{
    //! Type aliases
    using AtomicNumber = int;
    using ReadData     = std::function<ImportSBTable(AtomicNumber)>;
    using HostRef
        = CombinedBremData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = CombinedBremData<Ownership::const_reference, MemSpace::device>;
    //@}

  public:
    // Construct from model ID and other necessary data
    CombinedBremModel(ActionId              id,
                      const ParticleParams& particles,
                      const MaterialParams& materials,
                      ReadData              load_sb_table,
                      bool                  enable_lpm);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Apply the interaction kernel to host data
    void execute(CoreHostRef const&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreDeviceRef const&) const final;

    // ID of the model
    ActionId action_id() const final;

    //! Short name for the interaction kernel
    std::string label() const final { return "brems-combined"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "SB+relativistic electron+positron bremsstrahlung";
    }

    //! Access data on the host
    const HostRef& host_ref() const { return data_.host(); }

    //! Access data on the device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<CombinedBremData> data_;

    // Associated models
    std::shared_ptr<SeltzerBergerModel>    sb_model_;
    std::shared_ptr<RelativisticBremModel> rb_model_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
