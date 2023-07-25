//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/WentzelModel.cc
//---------------------------------------------------------------------------//
#include "WentzelModel.hh"

#include "celeritas_config.h"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/em/data/WentzelData.hh"
#include "celeritas/em/executor/WentzelExecutor.hh"
#include "celeritas/em/model/detail/MottInterpolatedCoefficients.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/io/ImportParameters.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID, particle data, and material data.
 */
WentzelModel::WentzelModel(ActionId id,
                           ParticleParams const& particles,
                           MaterialParams const& materials,
                           ImportEmParameters const& em_params,
                           SPConstImported data)
    : imported_(data,
                particles,
                ImportProcessClass::coulomb_scat,
                ImportModelClass::e_coulomb_scattering,
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(id);

    ScopedMem record_mem("WentzelModel.construct");
    HostVal<WentzelData> host_data;

    // This is where the data is built and transfered to the device
    host_data.ids.action = id;
    host_data.ids.electron = particles.find(pdg::electron());
    host_data.ids.positron = particles.find(pdg::positron());

    CELER_VALIDATE(host_data.ids,
                   << "missing IDs (required for " << this->description()
                   << ")");

    // Electron mass
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // TODO: Select form factor
    host_data.form_factor_type = NuclearFormFactorType::Exponential;

    // Thomas-Fermi constant C_TF
    const real_type ctf = fastpow(3 * constants::pi / 4, real_type{2} / 3) / 2;

    // Prefactor of the screen R squared
    host_data.screen_r_sq_elec = native_value_to<units::MevMomentumSq>(
        em_params.screening_factor
        * ipow<2>(constants::hbar_planck / (2 * ctf * constants::a0_bohr)));

    // This is the inverse of Geant's constn
    // need to multiply by 2 to match Geant's magic number
    host_data.form_momentum_scale = native_value_to<units::MevMomentumSq>(
        12 / ipow<2>(1.27 * (1e-15 * units::meter) / constants::hbar_planck)
        / 2);

    // Load Mott coefficients
    build_data(host_data, materials);

    // Construct data on device
    data_ = CollectionMirror<WentzelData>{std::move(host_data)};

    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto WentzelModel::applicability() const -> SetApplicability
{
    Applicability electron_applic;
    electron_applic.particle = this->host_ref().ids.electron;
    // TODO: construct actual energy range
    electron_applic.lower = zero_quantity();
    electron_applic.upper = max_quantity();

    Applicability positron_applic;
    positron_applic.particle = this->host_ref().ids.positron;
    positron_applic.lower = zero_quantity();
    positron_applic.upper = max_quantity();

    return {electron_applic, positron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto WentzelModel::micro_xs(Applicability applic) const -> MicroXsBuilders
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void WentzelModel::execute(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{WentzelExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void WentzelModel::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId WentzelModel::action_id() const
{
    return this->host_ref().ids.action;
}

//---------------------------------------------------------------------------//
/*!
 * Load Mott coefficients and construct per-element data.
 */
void WentzelModel::build_data(HostVal<WentzelData>& host_data,
                              MaterialParams const& materials)
{
    // Build element data
    unsigned int const num_elements = materials.num_elements();
    auto elem_data = make_builder(&host_data.elem_data);
    elem_data.reserve(num_elements);

    for (auto el_id : range(ElementId{num_elements}))
    {
        ElementView const& element = materials.get(el_id);
        const AtomicNumber z = element.atomic_number();

        WentzelElementData z_data;

        // Load Mott coefficients
        // Currently only support up to Z=92 (Uranium) as taken from Geant4
        int const index = (z.get() <= 92) ? z.get() : 0;
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                z_data.mott_coeff[i][j]
                    = detail::interpolated_mott_coeffs[index][i][j];
            }
        }

        elem_data.push_back(z_data);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
