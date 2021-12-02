//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RelativisticBremModel.cc
//---------------------------------------------------------------------------//
#include "RelativisticBremModel.hh"

#include "base/Assert.hh"
#include "base/Algorithms.hh"
#include "base/Constants.hh"

#include "base/Range.hh"
#include "base/CollectionBuilder.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/em/detail/RelativisticBremData.hh"
#include "physics/em/generated/RelativisticBremInteract.hh"

#include <cmath>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
RelativisticBremModel::RelativisticBremModel(ModelId               id,
                                             const ParticleParams& particles,
                                             const MaterialParams& materials,
                                             bool                  enable_lpm)
{
    CELER_EXPECT(id);

    HostValue host_ref;

    host_ref.ids.model    = id;
    host_ref.ids.electron = particles.find(pdg::electron());
    host_ref.ids.positron = particles.find(pdg::positron());
    host_ref.ids.gamma    = particles.find(pdg::gamma());

    CELER_VALIDATE(host_ref.ids,
                   << "missing IDs (required for " << this->label() << ")");

    // Save particle properties
    host_ref.electron_mass = particles.get(host_ref.ids.electron).mass();

    // Set the LPM flag (true by default)
    host_ref.enable_lpm = enable_lpm;

    // Build other data (host_ref.lpm_table, host_ref.elem_data))
    this->build_data(&host_ref, materials, host_ref.electron_mass.value());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<detail::RelativisticBremData>{std::move(host_ref)};
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
    electron_brem.lower    = this->host_ref().low_energy_limit();
    electron_brem.upper    = this->host_ref().high_energy_limit();

    Applicability positron_brem = electron_brem;
    positron_brem.particle      = this->host_ref().ids.positron;

    return {electron_brem, positron_brem};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void RelativisticBremModel::interact(const DeviceInteractRef& data) const
{
    generated::relativistic_brem_interact(this->device_ref(), data);
}

void RelativisticBremModel::interact(const HostInteractRef& data) const
{
    generated::relativistic_brem_interact(this->host_ref(), data);
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId RelativisticBremModel::model_id() const
{
    return this->host_ref().ids.model;
}

//---------------------------------------------------------------------------//
/*!
 * Build RelativisticBremData (lpm_table and elem_data).
 */
void RelativisticBremModel::build_data(HostValue*            data,
                                       const MaterialParams& materials,
                                       real_type             particle_mass)
{
    // Build a table of LPM functions, G(s) and \phi(s) in the range
    // s = [0, data->limit_s_lpm()] with the 1/data->inv_delta_lpm() interval
    auto         lpm_table  = make_builder(&data->lpm_table);
    unsigned int num_points = data->inv_delta_lpm() * data->limit_s_lpm() + 1;
    lpm_table.reserve(num_points);

    for (auto s_point : range(num_points))
    {
        auto shat   = static_cast<real_type>(s_point) / data->inv_delta_lpm();
        auto s_data = compute_lpm_data(shat);
        lpm_table.push_back(s_data);
    }

    // Build element data for available elements
    unsigned int num_elements = materials.num_elements();

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
 * Initialise data for a given element:
 * G4eBremsstrahlungRelModel::InitialiseElementData()
 */
auto RelativisticBremModel::compute_element_data(const ElementView& elem,
                                                 real_type electron_mass)
    -> ElementData
{
    ElementData data;

    AtomicNumber iz = min(elem.atomic_number(), 120);

    data.iz   = iz;
    data.logz = elem.log_z();

    real_type fc      = elem.coulomb_correction();
    real_type ff_el   = 1.0;
    real_type ff_inel = 1.0;

    data.logz = elem.log_z();
    data.fz   = data.logz / 3 + fc;

    if (iz < 5)
    {
        ff_el   = RelativisticBremModel::get_form_factor(iz).el;
        ff_inel = RelativisticBremModel::get_form_factor(iz).inel;
    }
    else
    {
        ff_el   = std::log(184.15) - data.logz / 3;
        ff_inel = std::log(1194.0) - 2 * data.logz / 3;
    }

    real_type z13 = elem.cbrt_z();
    real_type z23 = ipow<2>(z13);

    data.factor1        = (ff_el - fc) + ff_inel / iz;
    data.factor2        = (1 + real_type(1) / iz) / 12;
    data.s1             = z23 / ipow<2>(184.15);
    data.inv_logs1      = 1 / std::log(data.s1);
    data.inv_logs2      = 1 / (std::log(constants::sqrt_two * data.s1));
    data.gamma_factor   = 100 * electron_mass / z13;
    data.epsilon_factor = 100 * electron_mass / z23;

    return data;
}

//---------------------------------------------------------------------------//
/*!
 * Compute LPM function: G4eBremsstrahlungRelModel::ComputeLPMGsPhis
 */
auto RelativisticBremModel::compute_lpm_data(real_type x) -> MigdalData
{
    MigdalData data;

    if (x < 0.01)
    {
        data.phis = 6 * x * (1 - constants::pi * x);
        data.gs   = 12 * x - 2 * data.phis;
    }
    else
    {
        real_type x2 = ipow<2>(x);
        real_type x3 = x * x2;
        real_type x4 = ipow<2>(x2);

        // use Stanev approximation: for \psi(s) and compute G(s)
        if (x < 0.415827)
        {
            data.phis = 1
                        - std::exp(-6 * x * (1 + x * (3 - constants::pi))
                                   + x3 / (0.623 + 0.796 * x + 0.658 * x2));
            real_type psi = 1
                            - std::exp(-4 * x
                                       - 8 * x2
                                             / (1 + 3.936 * x + 4.97 * x2
                                                - 0.05 * x3 + 7.5 * x4));
            data.gs = 3 * psi - 2 * data.phis;
        }
        else if (x < 1.55)
        {
            data.phis = 1
                        - std::exp(-6 * x * (1 + x * (3 - constants::pi))
                                   + x3 / (0.623 + 0.796 * x + 0.658 * x2));
            data.gs = std::tanh(-0.160723 + 3.755030 * x - 1.798138 * x2
                                + 0.672827 * x3 - 0.120772 * x4);
        }
        else
        {
            data.phis = 1 - 0.011905 / x4;
            if (x < 1.9156)
            {
                data.gs = std::tanh(-0.160723 + 3.755030 * x - 1.798138 * x2
                                    + 0.672827 * x3 - 0.120772 * x4);
            }
            else
            {
                data.gs = 1 - 0.023065 / x4;
            }
        }
    }

    return data;
}

//---------------------------------------------------------------------------//
/*!
 * Elastic and inelatic form factor using the Dirac-Fock model of atom
 *
 * For light elements (Z < 5) where Thomas-Fermi model doesn't work.
 * Excerpted from G4eBremsstrahlungRelModel of Geant4 10.7.
 */
auto RelativisticBremModel::get_form_factor(AtomicNumber z) -> const FormFactor&
{
    CELER_EXPECT(z > 0 && z < 8);
    static const FormFactor form_factor[] = {{5.3104, 5.9173},
                                             {4.7935, 5.6125},
                                             {4.7402, 5.5377},
                                             {4.7112, 5.4728},
                                             {4.6694, 5.4174},
                                             {4.6134, 5.3688},
                                             {4.5520, 5.3236}};

    return form_factor[z - 1];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
