//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/model/DTMixMucfModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/mucf/data/DTMixMucfData.hh"
#include "celeritas/phys/Model.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion model for dd, dt, and tt molecules.
 *
 * In this model the executor performs the full muon-catalyzed fusion workflow.
 * It forms a muonic d or t atom, samples which muonic molecule will be
 * produced, selects the channel, and calls the appropriate interactor.
 *
 * The full set of "actions" is as follows, and in this ordering:
 * - Define muon decay time to compete with the rest of the execution
 * - Form muonic atom and select its spin
 * - May execute atom spin flip or atom transfer
 * - Form muonic molecule and select its spin
 * - Calculate mean cycle time (time it takes from atom formation to fusion)
 * - Confirm if fusion happens or the if the muon should decay
 * - Call appropriate Interactor: Muon decay, or one of the muCF interactors
 *
 * \note This is an at-rest model.
 */
class DTMixMucfModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    //! \name Type aliases
    using HostRef = HostCRef<DTMixMucfData>;
    using DeviceRef = DeviceCRef<DTMixMucfData>;
    //!@}

    // Construct from model ID and other necessary data
    DTMixMucfModel(ActionId id,
                   ParticleParams const& particles,
                   MaterialParams const& materials);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    XsTable micro_xs(Applicability) const final;

    // Apply the interaction kernel on host
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel on device
    void step(CoreParams const&, CoreStateDevice&) const final;

    //!@{
    //! Access model data
    HostRef const& host_ref() const { return data_.host_ref(); }
    DeviceRef const& device_ref() const { return data_.device_ref(); }
    //!@}

  private:
    // Host/device storage and reference
    CollectionMirror<DTMixMucfData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
