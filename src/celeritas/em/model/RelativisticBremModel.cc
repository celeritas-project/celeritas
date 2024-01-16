//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/RelativisticBremModel.cc
//---------------------------------------------------------------------------//
#include "RelativisticBremModel.hh"

#include <cmath>
#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/ElectronBremsData.hh"
#include "celeritas/em/data/RelativisticBremData.hh"
#include "celeritas/em/executor/RelativisticBremExecutor.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

#include "../interactor/detail/PhysicsConstants.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
RelativisticBremModel::RelativisticBremModel(ActionId id,
                                             ParticleParams const& particles,
                                             MaterialParams const& materials,
                                             SPConstImported data,
                                             bool enable_lpm)
    : imported_(data,
                particles,
                ImportProcessClass::e_brems,
                ImportModelClass::e_brems_lpm,
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(id);

    HostValue host_ref;

    host_ref.ids.action = id;
    host_ref.ids.electron = particles.find(pdg::electron());
    host_ref.ids.positron = particles.find(pdg::positron());
    host_ref.ids.gamma = particles.find(pdg::gamma());

    CELER_VALIDATE(host_ref.ids,
                   << "missing IDs (required for " << this->description()
                   << ")");

    // Save particle properties
    host_ref.electron_mass = particles.get(host_ref.ids.electron).mass();

    // Set the LPM flag (true by default)
    host_ref.enable_lpm = enable_lpm;

    // Build other data (host_ref.lpm_table, host_ref.elem_data))
    this->build_data(&host_ref, materials, host_ref.electron_mass.value());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<RelativisticBremData>{std::move(host_ref)};
    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto RelativisticBremModel::applicability() const -> SetApplicability
{
    Applicability electron_brem;
    electron_brem.particle = this->host_ref().ids.electron;
    electron_brem.lower = detail::seltzer_berger_limit();
    electron_brem.upper = detail::high_energy_limit();

    Applicability positron_brem = electron_brem;
    positron_brem.particle = this->host_ref().ids.positron;

    return {electron_brem, positron_brem};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto RelativisticBremModel::micro_xs(Applicability applic) const
    -> MicroXsBuilders
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Interact with host data.
 */
void RelativisticBremModel::execute(CoreParams const& params,
                                    CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{RelativisticBremExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void RelativisticBremModel::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId RelativisticBremModel::action_id() const
{
    return this->host_ref().ids.action;
}

//---------------------------------------------------------------------------//
/*!
 * Build RelativisticBremData (lpm_table and elem_data).
 */
void RelativisticBremModel::build_data(HostValue* data,
                                       MaterialParams const& materials,
                                       real_type particle_mass)
{
    // Build element data for available elements
    auto num_elements = materials.num_elements();

    auto elem_data = make_builder(&data->elem_data);
    elem_data.reserve(num_elements);

    for (auto el_id : range(ElementId{num_elements}))
    {
        auto z_data = compute_element_data(materials.get(el_id), particle_mass);
        elem_data.push_back(z_data);
    }
}
//---------------------------------------------------------------------------//
/*!
 * Initialise data for a given element.
 *
 * See \c G4eBremsstrahlungRelModel::InitialiseElementData() in Geant4.
 */
auto RelativisticBremModel::compute_element_data(ElementView const& elem,
                                                 real_type electron_mass)
    -> ElementData
{
    ElementData data;

    AtomicNumber z = min(elem.atomic_number(), AtomicNumber{120});

    real_type ff_el;
    real_type ff_inel;
    if (z < AtomicNumber{5})
    {
        ff_el = RelativisticBremModel::get_form_factor(z).el;
        ff_inel = RelativisticBremModel::get_form_factor(z).inel;
    }
    else
    {
        ff_el = real_type(std::log(184.15)) - elem.log_z() / 3;
        ff_inel = real_type(std::log(1194.0)) - 2 * elem.log_z() / 3;
    }

    real_type fc = elem.coulomb_correction();
    real_type invz = real_type(1) / z.unchecked_get();
    real_type z13 = elem.cbrt_z();
    real_type z23 = ipow<2>(z13);

    data.fz = elem.log_z() / 3 + fc;
    data.factor1 = (ff_el - fc) + ff_inel * invz;
    data.factor2 = (1 + invz) / 12;
    data.gamma_factor = 100 * electron_mass / z13;
    data.epsilon_factor = 100 * electron_mass / z23;

    return data;
}

//---------------------------------------------------------------------------//
/*!
 * Elastic and inelatic form factor using the Dirac-Fock model of atom.
 *
 * For light elements (Z < 5) where Thomas-Fermi model doesn't work.
 * Excerpted from G4eBremsstrahlungRelModel of Geant4 10.7.
 */
auto RelativisticBremModel::get_form_factor(AtomicNumber z) -> FormFactor const&
{
    CELER_EXPECT(z && z < AtomicNumber{8});
    static FormFactor const form_factor[] = {{5.3104, 5.9173},
                                             {4.7935, 5.6125},
                                             {4.7402, 5.5377},
                                             {4.7112, 5.4728},
                                             {4.6694, 5.4174},
                                             {4.6134, 5.3688},
                                             {4.5520, 5.3236}};

    return form_factor[z.unchecked_get() - 1];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
