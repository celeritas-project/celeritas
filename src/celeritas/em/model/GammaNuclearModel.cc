//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/GammaNuclearModel.cc
//---------------------------------------------------------------------------//
#include "GammaNuclearModel.hh"

#include "corecel/math/Quantity.hh"
#include "celeritas/g4/EmExtraPhysicsHelper.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
GammaNuclearModel::GammaNuclearModel(ActionId id,
                                     ParticleParams const& particles,
                                     MaterialParams const& materials,
                                     ReadData load_data)
    : StaticConcreteAction(id, "gamma-nuclear", "interact by gamma-nuclear")
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    HostVal<GammaNuclearData> data;

    helper_ = std::make_shared<EmExtraPhysicsHelper>();

    // Save IDs
    data.scalars.gamma_id = particles.find(pdg::gamma());

    CELER_VALIDATE(data.scalars.gamma_id,
                   << "missing gamma (required for " << this->description()
                   << ")");

    // Save particle properties
    CELER_EXPECT(data.scalars);

    // Load gamma-nuclear element cross section data
    NonuniformGridInserter insert_micro_xs_iaea{&data.reals_iaea,
                                                &data.micro_xs_iaea};
    NonuniformGridInserter insert_micro_xs_chips{&data.reals_chips,
                                                 &data.micro_xs_chips};

    double const emax = data.scalars.max_valid_energy().value();
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        // Build element cross sections from G4PARTICLEXS
        insert_micro_xs_iaea(load_data(z));

        // Build element cross sections above the upper bound of G4PARTICLEXS
        double emin = data.reals_iaea[data.micro_xs_iaea[el_id].grid.back()];
        insert_micro_xs_chips(this->calc_chips_xs(z, emin, emax));
    }
    CELER_ASSERT(data.micro_xs_iaea.size() == materials.num_elements());
    CELER_ASSERT(data.micro_xs_iaea.size() == data.micro_xs_chips.size());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<GammaNuclearData>{std::move(data)};
    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto GammaNuclearModel::applicability() const -> SetApplicability
{
    Applicability gamma_nuclear_applic;
    gamma_nuclear_applic.particle = this->host_ref().scalars.gamma_id;
    gamma_nuclear_applic.lower = zero_quantity();
    gamma_nuclear_applic.upper = this->host_ref().scalars.max_valid_energy();

    return {gamma_nuclear_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto GammaNuclearModel::micro_xs(Applicability) const -> XsTable
{
    // Cross sections are calculated on the fly
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void GammaNuclearModel::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_IMPLEMENTED("Gamma-nuclear inelastic interaction");
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void GammaNuclearModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Build CHIPS gamma-nuclear element cross sections using G4GammaNuclearXS
 * above the upper energy limit of G4PARTICLEXS/gamma (IAEA) data. The cross
 * sections are derived from the parameterization developed by M. V. Kossov
 * (CERN/ITEP Moscow) in the high energy region (106 MeV < E < 50 GeV) and
 * from a Reggeon-based parameterization in the ultra-high-energy region
 * (E > 50 GeV). G4GammaNuclearXS uses CHIPS (G4PhotoNuclearCrossSection)
 * above 150 MeV and performs linear interpolation between the upper energy
 * limit of G4PARTICLEXS/gamma (IAEA) data and 150 MeV.
 */
GammaNuclearModel::result_type
GammaNuclearModel::calc_chips_xs(AtomicNumber atomic_number,
                                 double emin,
                                 double emax) const
{
    CELER_EXPECT(atomic_number);

    result_type result;

    // Tabulate cross sections using separate parameterizations for the high
    // energy region (emin < E < 50 GeV) and the ultra high energy region up
    // to the maximum valid energy (emax). The numbers of bins are chosen to
    // adequately capture both the parameterized points (224 bins from 106 MeV
    // to 50 GeV) and the calculations used in G4PhotoNuclearCrossSection,
    // which can be made configurable if needed (TODO). Note that the linear
    // interpolation between the upper limit of the IAEA cross-section data
    // and 150 MeV, as used in G4GammaNuclearXS, is also included in this
    // tabulation.
    double const emid = helper_->max_high_energy();
    size_type nbin_high = 250;
    size_type nbin_ultra = 50;

    result.x.resize(nbin_high + nbin_ultra);
    result.y.resize(nbin_high + nbin_ultra);

    // Bin sizes in logarithmic energy scale
    double const de_high = (std::log(emid) - std::log(emin)) / (nbin_high - 1);
    double const de_ultra = (std::log(emax) - std::log(emid))
                            / (nbin_ultra - 1);

    // Tabulate the cross section from emin to emax
    MmSqMicroXs xs;
    for (size_type i = 0; i < nbin_high + nbin_ultra; ++i)
    {
        double energy
            = (i < nbin_high)
                  ? std::exp(std::log(emin) + de_high * i)
                  : std::exp(std::log(emid) + de_ultra * (i - nbin_high + 1));

        result.x[i] = energy;
        xs.value()
            = helper_->GammaNuclearElementXS(energy, atomic_number.get());
        result.y[i] = native_value_to<BarnXs>(native_value_from(xs)).value();
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
