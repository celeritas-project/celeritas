//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/GammaNuclearModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>

#include "corecel/data/ParamsDataStore.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/GammaNuclearData.hh"
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
 * Set up and launch the gamma-nuclear model interaction.
 *
 * The class also builds element cross-section tables using G4PARTICLEXS/gamma
 * (IAEA) data at low energies and CHIPS gamma–nuclear cross sections using
 * G4GammaNuclearXS above the IAEA upper energy limit (~130 MeV). The CHIPS
 * cross sections are based on the parameterization developed by M. V. Kossov
 * (CERN/ITEP Moscow) for the high energy region (106 MeV < E < 50 GeV) and on
 * a Reggeon-based parameterization for the ultra high energy region
 * (E > 50 GeV), as described in
 * \citet{degtyarenko-chiralinvariant-2000,
 * https://doi.org/10.1007/s100500070026}.
 * G4GammaNuclearXS uses CHIPS (G4PhotoNuclearCrossSection) above 150 MeV and
 * performs linear interpolation between the upper limit of the G4PARTICLEXS
 * gamma-nuclear (IAEA) data and 150 MeV.
 */
class GammaNuclearModel final : public Model, public StaticConcreteAction
{
  public:
    //!@{
    using MevEnergy = units::MevEnergy;
    using ReadData = std::function<inp::Grid(AtomicNumber)>;
    using HostRef = HostCRef<GammaNuclearData>;
    using DeviceRef = DeviceCRef<GammaNuclearData>;
    //!@}

  public:
    // Construct from model ID and other necessary data
    GammaNuclearModel(ActionId id,
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
    HostRef const& host_ref() const { return data_.host_ref(); }
    DeviceRef const& device_ref() const { return data_.device_ref(); }
    //!@}

  private:
    //// DATA ////
    std::shared_ptr<EmExtraPhysicsHelper> helper_;

    // Host/device storage and reference
    ParamsDataStore<GammaNuclearData> data_;

    //// TYPES ////

    using HostXsData = HostVal<GammaNuclearData>;

    //// HELPER FUNCTIONS ////

    inp::Grid
    calc_chips_xs(AtomicNumber atomic_number, double emin, double emax) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
