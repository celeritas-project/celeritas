//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/ChipsNeutronElasticModel.cc
//---------------------------------------------------------------------------//
#include "ChipsNeutronElasticModel.hh"

#include "corecel/math/Quantity.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/neutron/executor/ChipsNeutronElasticExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
ChipsNeutronElasticModel::ChipsNeutronElasticModel(
    ActionId id,
    ParticleParams const& particles,
    MaterialParams const& materials,
    ReadData load_data)
    : StaticConcreteAction(id,
                           "neutron-elastic-chips",
                           "interact by neutron elastic scattering (CHIPS)")
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    HostVal<NeutronElasticData> data;

    // Save IDs
    data.neutron = particles.find(pdg::neutron());

    CELER_VALIDATE(data.neutron,
                   << "missing neutron particles (required for "
                   << this->description() << ")");

    // Save particle properties
    data.neutron_mass = particles.get(data.neutron).mass();

    // Load neutron elastic cross section data
    NonuniformGridInserter insert_grid{&data.reals, &data.micro_xs};
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        insert_grid(load_data(z));
    }
    CELER_ASSERT(data.micro_xs.size() == materials.num_elements());

    // Add A(Z,N)-dependent coefficients of the CHIPS elastic interaction
    make_builder(&data.coeffs).reserve(materials.num_isotopes());
    for (auto iso_id : range(IsotopeId{materials.num_isotopes()}))
    {
        AtomicMassNumber a = materials.get(iso_id).atomic_mass_number();
        this->append_coeffs(a, &data);
    }
    CELER_ASSERT(data.coeffs.size() == materials.num_isotopes());

    // Move to mirrored data, copying to device
    mirror_ = ParamsDataStore<NeutronElasticData>{std::move(data)};
    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto ChipsNeutronElasticModel::applicability() const -> SetApplicability
{
    Applicability neutron_applic;
    neutron_applic.particle = this->host_ref().neutron;
    neutron_applic.lower = this->host_ref().min_valid_energy();
    neutron_applic.upper = this->host_ref().max_valid_energy();

    return {neutron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto ChipsNeutronElasticModel::micro_xs(Applicability) const -> XsTable
{
    // Cross sections are calculated on the fly
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void ChipsNeutronElasticModel::step(CoreParams const& params,
                                    CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{ChipsNeutronElasticExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ChipsNeutronElasticModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Construct A-dependent coefficients of the CHIPS differential cross section
 * for a single isotope used in the G4ChipsNeutronElasticXS class.
 *
 * \note The parameter coeffs[index] is equivalent to lastPAR[index + 15] used
 * in G4ChipsNeutronElasticXS::GetChipsCrossSection. The first 15 parameters
 * of lastPAR are not used for sampling the momentum transfer between neutron
 * and nucleus (Z > 0, N >= 1).
 *
 * M.V. Kossov, “Manual for the CHIPS event generator”, KEK internal report
 * 2000-17, Feb. 2001 H/R; P.V. Degtyarenko, M.V. Kossov and H.P. Wellisch,
 * Eur. Phys. J. A8, 217 (2000); P.V. Degtyarenko, M.V. Kossov and H.P.
 * Wellisch, Eur. Phys. J. A9 (2001).
 */
void ChipsNeutronElasticModel::append_coeffs(AtomicMassNumber amass,
                                             HostXsData* data) const
{
    ChipsDiffXsCoefficients coeffs;

    real_type a = static_cast<real_type>(amass.get());

    real_type sa = std::sqrt(a);
    real_type asa = a * sa;
    real_type a2 = ipow<2>(a);
    real_type a3 = a2 * a;
    real_type a4 = a3 * a;
    real_type a5 = a4 * a;
    real_type a6 = a4 * a2;
    real_type a7 = a6 * a;
    real_type a8 = a7 * a;
    real_type a9 = a8 * a;
    real_type a10 = a9 * a;
    real_type a12 = ipow<2>(a6);
    real_type a14 = ipow<2>(a7);
    real_type a16 = ipow<2>(a8);
    real_type a32 = ipow<2>(a16);

    // The differential cross-section is parameterized separately for
    // A>6 & A<7
    if (amass <= AtomicMassNumber{6})
    {
        // The main pre-exponent (pel_sg)
        coeffs.par[0] = 4e3 * a;
        coeffs.par[1] = 1.2e7 * a8 + 380 * a16 * a;
        coeffs.par[2] = 0.7 / (1 + 4e-12 * a16);
        coeffs.par[3] = 2.5 / a8 / (a4 + 1e-16 * a32);
        coeffs.par[4] = 0.28 * a;
        coeffs.par[5] = 1.2 * a2 + 2.3;
        coeffs.par[6] = 3.8 / a;
        // The main slope (pel_sl)
        coeffs.par[7] = 0.01 / (1 + 2.4e-3 * a5);
        coeffs.par[8] = 0.2 * a;
        coeffs.par[9] = 9e-7 / (1 + 0.035 * a5);
        coeffs.par[10] = (42 + 2.7e-11 * a16) / (1 + 0.14 * a);
        // The main quadratic (pel_sh)
        coeffs.par[11] = 2.25 * a3;
        coeffs.par[12] = 18;
        coeffs.par[13] = 2.4e-3 * a8 / (1 + 2.6e-4 * a7);
        coeffs.par[14] = 3.5e-36 * a32 * a8 / (1 + 5e-15 * a32 / a);
        coeffs.par[15] = 1e5 / (a8 + 2.5e12 / a16);
        coeffs.par[16] = 8e7 / (a12 + 1e-27 * ipow<2>(a16 * a12));
        coeffs.par[17] = 6e-4 * a3;
        // The 1st max slope (pel_qs)
        coeffs.par[18] = 10 + 4e-8 * a12 * a;
        coeffs.par[19] = 0.114;
        coeffs.par[20] = 3e-3;
        coeffs.par[21] = 2e-23;
        // The effective pre-exponent (pel_ss)
        coeffs.par[22] = 1 / (1 + 1e-4 * a8);
        coeffs.par[23] = 1.5e-4 / (1 + 5e-6 * a12);
        coeffs.par[24] = 0.03;
        // The effective slope (pel_sb)
        coeffs.par[25] = a / 2;
        coeffs.par[26] = 2e-7 * a4;
        coeffs.par[27] = 4;
        coeffs.par[28] = 64 / a3;
        // The gloria pre-exponent (pel_us)
        coeffs.par[29] = 1e8 * std::exp(0.32 * asa);
        coeffs.par[30] = 20 * std::exp(0.45 * asa);
        coeffs.par[31] = 7e3 + 2.4e6 / a5;
        coeffs.par[32] = 2.5e5 * std::exp(0.085 * a3);
        coeffs.par[33] = 2.5 * a;
        // The gloria slope (pel_ub)
        coeffs.par[34] = 920 + 0.03 * a8 * a3;
        coeffs.par[35] = 93 + 2.3e-3 * a12;
    }
    else
    {
        // The main pre-exponent (peh_sg)
        coeffs.par[0] = 4.5 * std::pow(a, 1.15);
        coeffs.par[1] = 0.06 * std::pow(a, 0.6);
        coeffs.par[2] = 0.6 * a / (1 + 2e15 / a16);
        coeffs.par[3] = 0.17 / (a + 9e5 / a3 + 1.5e33 / a32);
        coeffs.par[4] = (1e-3 + 7e-11 * a5) / (1 + 4.4e-11 * a5);
        coeffs.par[5] = (ipow<2>(2.2e-28 * a10) + 2e-29) / (1 + 2e-22 * a12);
        // The main slope (peh_sl)
        coeffs.par[6] = 400 / a12 + 2e-22 * a9;
        coeffs.par[7] = 1e-32 * a12 / (1 + 5e22 / a14);
        coeffs.par[8] = 1e3 / a2 + 9.5 * sa * std::sqrt(sa);
        coeffs.par[9] = 4e-6 * a * asa + 1e11 / a16;
        coeffs.par[10] = (120 / a + 2e-3 * a2) / (1 + 2e14 / a16);
        coeffs.par[11] = 9 + 100 / a;
        // The main quadratic (peh_sh)
        coeffs.par[12] = 2e-3 * a3 + 3e7 / a6;
        coeffs.par[13] = 7e-15 * a4 * asa;
        coeffs.par[14] = 9e3 / a4;
        // The 1st max pre-exponent (peh_qq)
        coeffs.par[15] = 1.1e-3 * asa / (1 + 3e34 / a32 / a4);
        coeffs.par[16] = 1e-5 * a2 + 2e14 / a16;
        coeffs.par[17] = 1.2e-11 * a2 / (1 + 1.5e19 / a12);
        coeffs.par[18] = 0.016 * asa / (1 + 5e16 / a16);
        // The 1st max slope (peh_qs)
        coeffs.par[19] = 2e-3 * a4 / (1 + 7e7 / std::pow(a - 6.83, 14));
        coeffs.par[20] = 2e6 / a6 + 7.2 / std::pow(a, 0.11);
        coeffs.par[21] = 11 * a3 / (1 + 7e23 / a16 / a8);
        coeffs.par[22] = 100 / asa;
        // The 2nd max pre-exponent (peh_ss)
        coeffs.par[23] = (0.1 + 4.4e-5 * a2) / (1 + 5e5 / a4);
        coeffs.par[24] = 3.5e-4 * a2 / (1 + 1e8 / a8);
        coeffs.par[25] = 1.3 + 3e5 / a4;
        coeffs.par[26] = 500 / (a2 + 50) + 3;
        coeffs.par[27] = 1e-9 / a + ipow<4>(6e14 / a16);
        // The 2nd max slope (peh_sb)
        coeffs.par[28] = 0.4 * asa + 3e-9 * a6;
        coeffs.par[29] = 5e-4 * a5;
        coeffs.par[30] = 2e-3 * a5;
        coeffs.par[31] = 10;  // p4
        // The effective pre-exponent (peh_us)
        coeffs.par[32] = 0.05 + 5e-3 * a;
        coeffs.par[33] = 7e-8 / sa;
        coeffs.par[34] = 0.8 * sa;
        coeffs.par[35] = 0.02 * sa;
        coeffs.par[36] = 1e8 / a3;
        coeffs.par[37] = 3e32 / (a32 + 1e32);
        // The effective slope (peh_ub)
        coeffs.par[38] = 24;
        coeffs.par[39] = 20 / sa;
        coeffs.par[40] = 7e3 * a / (sa + 1);
        coeffs.par[41] = 900 * sa / (1 + 500 / a3);
    }

    make_builder(&data->coeffs).push_back(coeffs);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
