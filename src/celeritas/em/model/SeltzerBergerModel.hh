//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/SeltzerBergerModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/SeltzerBergerData.hh"
#include "celeritas/io/ImportSBTable.hh"
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
 * Manage the Seltzer-Berger model for Bremsstrahlung.
 *
 * The total scaled bremsstrahlung differential cross section for an element Z
 * is defined as
 * \f[
 *   \chi(Z,E,\kappa) = \frac{\beta^2}{Z^2} k \difd{\sigma}{k},
 * \f]
 * where \f$ \kappa = k / E \f$ is the ratio of the emitted photon energy to
 * the incident charged particle energy, \f$ \beta \f$ is the ratio of the
 * charged particle velocity to the speed of light, and
 * \f$ \difd{\sigma}{k} \f$ is the bremsstrahlung differential cross
 * section.
 *
 * Seltzer and Berger have tabulated the scaled DCS (in mb) for elements Z = 1
 * - 100 and for incident charged particle energies from 1 keV to 10 GeV
 * (reported in MeV) in Seltzer S.M. and M.J. Berger (1986), "Bremsstrahlung
 * energy spectra from electrons with kinetic energy 1 keV–10 GeV incident on
 * screened nuclei and orbital electrons of neutral atoms with Z = 1–100", At.
 * Data Nucl. Data Tables 35, 345–418.
 */
class SeltzerBergerModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    using Mass = units::MevMass;
    using ReadData = std::function<ImportSBTable(AtomicNumber)>;
    using HostRef = HostCRef<SeltzerBergerData>;
    using DeviceRef = DeviceCRef<SeltzerBergerData>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    SeltzerBergerModel(ActionId id,
                       ParticleParams const& particles,
                       MaterialParams const& materials,
                       SPConstImported data,
                       ReadData load_sb_table);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    // Get the microscopic cross sections for the given particle and material
    XsTable micro_xs(Applicability) const final;

    // Apply the interaction kernel on device
    void step(CoreParams const&, CoreStateHost&) const final;

    // Apply the interaction kernel
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Access SB data on the host
    HostRef const& host_ref() const { return data_.host_ref(); }

    //! Access SB data on the device
    DeviceRef const& device_ref() const { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<SeltzerBergerData> data_;

    ImportedModelAdapter imported_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
